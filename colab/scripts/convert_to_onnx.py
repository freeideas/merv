#!/usr/bin/env python
"""Convert the merged Mervin/Mervis Phi-4-mini model to ONNX q4 for Transformers.js.

Colab edition of the VM script (scripts/convert_to_onnx.py). Same pipeline, but
paths are CLI args and `optimum-cli` is found next to whatever Python runs this
(so it works inside the throwaway conversion venv the notebook builds).

Pipeline:
  1. optimum-cli export onnx  (fp32, with KV-cache)        -> WORK/model.onnx
  2. 4-bit weight-only quantize (MatMulNBits, block 32, symmetric, QOperator);
     weights only, activations stay fp32                    -> OUT/onnx/model_q4.onnx
  2b. fp16 the fp32 token-embedding table (post-Gather Cast back to fp32) so the
     model fits under ORT-Web's ~4 GB WASM load ceiling     -> ~4.86 GB -> ~3.63 GB
  3. copy tokenizer/config into OUT/, make the outputs world-readable

Why q4 and NOT q4f16 (the lesson that cost the VM team real time): casting the
graph with onnxconverter_common.convert_float_to_float16 half-converts the fp32
island that Phi-3's RMSNorm deliberately upcasts into, leaving a layernorm `Add`
with one fp32 and one fp16 operand. onnxruntime then refuses to load the model
("Type parameter (T) of Optype (Add) bound to different types"). Reordering
cast-vs-quantize does NOT help -- the cast itself is the problem. So we ship q4
(4-bit weights, fp32 activations) and instead do ONE surgical fp16 cast on just
the embedding table (see fp16_embedding_cast): that alone drops ~4.86 GB ->
~3.63 GB, enough to clear ORT-Web's ~4 GB 32-bit WASM heap (the whole model is
loaded into WASM before WebGPU gets it). A full q4f16 (~3.4 GB) would need
explicit float16 op/node block-lists around the RMSNorm region -- a later
optimization; the embedding cast gets most of the win without that risk.

Browser side must match: app.js uses `dtype:'q4'` and Transformers.js feeds
**fp32** past_key_values (a q4f16 build would want fp16).

Usage (run with the conversion venv's python):
  /content/convenv/bin/python convert_to_onnx.py SRC OUT
    SRC = merged HF model dir   (default: /content/mervis-merged)
    OUT = transformers.js dir   (default: /content/web/model)

If memory is tight on a free T4 (the fp32 export is ~17.8 GB on disk and RAM
hungry), see the FP16_GPU_EXPORT note at the bottom for a lighter alternative.
"""
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/mervis-merged")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/content/web/model")
WORK = Path("/content/onnx_fp32")
ONNX_OUT = OUT / "onnx"
BIN = Path(sys.executable).parent  # the venv's bin/ -> has optimum-cli

TOKENIZER_FILES = [
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json",
]


def export_fp32():
    if (WORK / "model.onnx").exists():
        print("[1/3] fp32 ONNX already exists -> skipping export")
        return
    print("[1/3] exporting fp32 ONNX (text-generation-with-past, CPU)...", flush=True)
    subprocess.run(
        [
            str(BIN / "optimum-cli"), "export", "onnx",
            "--model", str(SRC),
            "--task", "text-generation-with-past",
            "--framework", "pt",
            str(WORK),
        ],
        check=True,
    )


def fp16_embedding_cast(model):
    """Shrink the model under the ~4 GB ONNX-Runtime-Web WASM load ceiling.

    ORT-Web loads the whole model into a 32-bit WASM heap (~4 GB cap) *before*
    handing tensors to WebGPU, so a 4.86 GB q4 build won't load. MatMulNBits only
    quantizes MatMul ops, so the token-embedding table ([vocab, hidden] ~=
    200064x3072) stays fp32 -- ~2.46 GB, over half the file. Cast that one
    initializer to fp16 and Cast the Gather output back to fp32, so every
    downstream consumer (incl. Phi-3's fp32 RMSNorm island) still sees fp32.
    Net: ~4.86 GB -> ~3.63 GB, q4 semantics (and dtype:'q4' + fp32 KV) unchanged.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    g = model.graph
    inits = {i.name: i for i in g.initializer}
    cands = []
    for node in g.node:
        if node.op_type != "Gather":
            continue
        init = inits.get(node.input[0])
        if init is None or init.data_type != TensorProto.FLOAT:
            continue
        dims = list(init.dims)
        # embedding table: 2D, vocab-sized rows, hidden-sized cols. The >=1024
        # col guard rejects the rotary cos/sin caches (head_dim ~128 cols).
        if len(dims) == 2 and dims[0] >= 50000 and dims[1] >= 1024:
            cands.append((int(np.prod(dims)), node, init, dims))
    if not cands:
        print("      [fp16-emb] no fp32 embedding Gather found -> skipping")
        return model
    cands.sort(key=lambda x: x[0], reverse=True)
    _, emb_node, emb_init, dims = cands[0]

    arr = numpy_helper.to_array(emb_init).astype(np.float16)
    emb_init.CopyFrom(numpy_helper.from_array(arr, emb_init.name))

    gout = emb_node.output[0]
    cast_out = gout + "_fp32"
    for n in g.node:
        if n is emb_node:
            continue
        for i, inp in enumerate(n.input):
            if inp == gout:
                n.input[i] = cast_out
    for o in g.output:
        if o.name == gout:
            o.name = cast_out
    cast = helper.make_node("Cast", [gout], [cast_out],
                            to=TensorProto.FLOAT, name=emb_node.name + "_to_fp32")
    g.node.insert(list(g.node).index(emb_node) + 1, cast)
    print(f"      [fp16-emb] {emb_init.name} {dims} fp32->fp16 + post-Gather Cast")
    return model


SHARD_MAX = 1_700_000_000  # keep each external-data file under V8's ~2 GB cap


def save_sharded(model, onnx_dir, model_filename, data_prefix, shard_max=SHARD_MAX):
    """Save the ONNX graph with external data split into <2 GB shards + a manifest.

    ORT-Web's default external-data path fetches the whole *.onnx_data into ONE
    Uint8Array, and V8 caps a single ArrayBuffer at ~2 GB -- so a 3.63 GB sidecar
    can never load (the 4 GB WASM *heap* is fine; the 2 GB *buffer* is the wall).
    We bin-pack the initializers into shards under that cap and emit
    external_data_manifest.json, which the browser feeds to ORT as
    session_options.externalData so each shard is fetched as its own buffer.
    """
    import json
    import os
    import onnx
    from onnx.external_data_helper import set_external_data

    onnx_dir = Path(onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    for p in onnx_dir.glob(f"{data_prefix}*"):  # clear prior single-file / shards
        p.unlink()
    for name in (model_filename, "external_data_manifest.json"):
        if (onnx_dir / name).exists():
            (onnx_dir / name).unlink()

    shards, off, fh = [], 0, None

    def new_shard():
        nonlocal fh, off
        if fh is not None:
            fh.close()
        name = f"{data_prefix}_{len(shards)}"
        shards.append(name)
        off = 0
        return open(onnx_dir / name, "wb")

    fh = new_shard()
    for t in model.graph.initializer:
        if not t.HasField("raw_data"):
            continue
        blob = t.raw_data
        if off > 0 and off + len(blob) > shard_max:
            fh = new_shard()
        fh.write(blob)
        # set_external_data reads raw_data, so call it BEFORE clearing it; it also
        # sets data_location = EXTERNAL for us.
        set_external_data(t, location=shards[-1], offset=off, length=len(blob))
        t.ClearField("raw_data")
        off += len(blob)
    if fh is not None:
        fh.close()

    onnx.save_model(model, str(onnx_dir / model_filename), save_as_external_data=False)
    (onnx_dir / "external_data_manifest.json").write_text(
        json.dumps({"model": model_filename, "shards": shards}, indent=1))
    for name in shards + [model_filename, "external_data_manifest.json"]:
        os.chmod(onnx_dir / name, 0o644)  # Caddy runs as another user; 0600 -> 403
    total = sum((onnx_dir / s).stat().st_size for s in shards) / 1e9
    print(f"      sharded external data -> {len(shards)} files, {total:.2f} GB total")
    for s in shards:
        print(f"        {s}  ({(onnx_dir / s).stat().st_size/1e9:.2f} GB)")


def quantize():
    import os
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer,
        QuantFormat,
    )

    src = WORK / "model.onnx"
    print(f"[2/3] loading {src} ...", flush=True)
    model = onnx.load(str(src), load_external_data=True)

    # NOTE: deliberately NO fp16 cast. convert_float_to_float16 half-converts
    # Phi-3's RMSNorm fp32 island -> a layernorm Add with mixed fp32/fp16
    # operands that onnxruntime refuses to load. We ship q4 (fp32 activations).
    # See the module docstring for the full story.
    print("      4-bit weight-only quantize (block_size=32, symmetric)...", flush=True)
    quant = MatMulNBitsQuantizer(
        model,
        bits=4,
        block_size=32,
        is_symmetric=True,
        quant_format=QuantFormat.QOperator,
    )
    quant.process()
    qmodel = quant.model.model if hasattr(quant.model, "model") else quant.model

    # fp16 the big fp32 embedding table so the model fits under ORT-Web's ~4 GB
    # WASM load ceiling (4.86 -> ~3.63 GB). See fp16_embedding_cast() above.
    print("      fp16 embedding cast (shrink under the 4 GB WASM ceiling)...", flush=True)
    qmodel = fp16_embedding_cast(qmodel)

    print(f"[3/3] saving {ONNX_OUT / 'model_q4.onnx'} (sharded external data)...",
          flush=True)
    save_sharded(qmodel, ONNX_OUT, "model_q4.onnx", "model_q4.onnx_data")


def assemble():
    OUT.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in TOKENIZER_FILES:
        s = SRC / name
        if s.exists():
            shutil.copy2(s, OUT / name)
            copied.append(name)
    print("      copied tokenizer/config:", ", ".join(copied))


def main():
    export_fp32()
    quantize()
    assemble()
    size = sum(f.stat().st_size for f in ONNX_OUT.glob("*")) / 1e9
    print(f"\nDONE. {ONNX_OUT} total = {size:.2f} GB")
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT)}  ({f.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())

# -----------------------------------------------------------------------------
# FP16_GPU_EXPORT (untested) -- a lighter export AND a possible *working* q4f16.
# Exporting straight to fp16 on the GPU skips the 17.8 GB fp32 monster (fp16 ONNX
# is ~7.6 GB). Crucially, optimum builds a *natively*-typed fp16 graph rather than
# the post-hoc convert_float_to_float16 cast that breaks Phi-3's RMSNorm -- so
# quantizing THAT might finally yield a loadable q4f16 (~3.4 GB). Drop-in for
# export_fp32():
#
#   subprocess.run([str(BIN/"optimum-cli"), "export", "onnx",
#       "--model", str(SRC), "--task", "text-generation-with-past",
#       "--device", "cuda", "--dtype", "fp16", str(WORK)], check=True)
#
# If you try it: rename the outputs back to model_q4f16.onnx, and the browser must
# switch to dtype:'q4f16' with an fp16 KV cache.
# -----------------------------------------------------------------------------
