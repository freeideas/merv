#!/usr/bin/env python
"""Convert the merged Mervin/Mervis Phi-4-mini model to ONNX q4 for Transformers.js.

Colab edition of the VM script (scripts/convert_to_onnx.py). Same pipeline, but
paths are CLI args and `optimum-cli` is found next to whatever Python runs this
(so it works inside the throwaway conversion venv the notebook builds).

Pipeline:
  1. optimum-cli export onnx  (fp32, with KV-cache)        -> WORK/model.onnx
  2. 4-bit weight-only quantize (MatMulNBits, block 32, symmetric, QOperator);
     weights only, activations stay fp32                    -> OUT/onnx/model_q4.onnx
  3. copy tokenizer/config into OUT/, make the outputs world-readable

Why q4 and NOT q4f16 (the lesson that cost the VM team real time): casting the
graph with onnxconverter_common.convert_float_to_float16 half-converts the fp32
island that Phi-3's RMSNorm deliberately upcasts into, leaving a layernorm `Add`
with one fp32 and one fp16 operand. onnxruntime then refuses to load the model
("Type parameter (T) of Optype (Add) bound to different types"). Reordering
cast-vs-quantize does NOT help -- the cast itself is the problem. So we ship q4
(4-bit weights, fp32 activations): ~4.9 GB vs ~3.4 GB and a touch slower, but it
loads and runs. A real q4f16 needs explicit float16 op/node block-lists around
the RMSNorm region -- a later size optimization.

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

    ONNX_OUT.mkdir(parents=True, exist_ok=True)
    out = ONNX_OUT / "model_q4.onnx"
    data = ONNX_OUT / "model_q4.onnx_data"
    print(f"[3/3] saving {out} (+ external data)...", flush=True)
    for p in (out, data):
        if p.exists():
            p.unlink()
    onnx.save(
        qmodel,
        str(out),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model_q4.onnx_data",
        convert_attribute=False,
    )
    # onnx.save writes the big *.onnx_data as 0600; Caddy runs as another user
    # and would 403 on it. Make every output world-readable.
    for p in (out, data):
        os.chmod(p, 0o644)


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
