#!/usr/bin/env python
"""Convert the merged Mervin/Mervis Phi-4-mini model to ONNX q4f16 for Transformers.js.

Colab edition of the VM script (scripts/convert_to_onnx.py). Same pipeline, but
paths are CLI args and `optimum-cli` is found next to whatever Python runs this
(so it works inside the throwaway conversion venv the notebook builds).

Pipeline:
  1. optimum-cli export onnx  (fp32, with KV-cache)        -> WORK/model.onnx
  2. cast fp32 -> fp16, THEN 4-bit weight-only quantize
     (MatMulNBits, block 32, symmetric, QOperator)         -> OUT/onnx/model_q4f16.onnx
  3. copy tokenizer/config into OUT/

Order note (the bit that took the VM team real time): cast the *clean* fp32
graph to fp16 FIRST (uniform, no mixed types), THEN 4-bit quantize. Doing it the
other way leaves MatMulNBits nodes that convert_float_to_float16 mishandles ->
mixed fp32/fp16 inputs that ORT-web refuses to run.

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
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer,
        QuantFormat,
    )
    from onnxconverter_common import float16

    src = WORK / "model.onnx"
    print(f"[2/3] loading {src} ...", flush=True)
    model = onnx.load(str(src), load_external_data=True)

    print("      casting fp32 -> fp16 (truncation warnings are expected)...", flush=True)
    model = float16.convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=True
    )

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
    out = ONNX_OUT / "model_q4f16.onnx"
    print(f"[3/3] saving {out} (+ external data)...", flush=True)
    for p in (out, ONNX_OUT / "model_q4f16.onnx_data"):
        if p.exists():
            p.unlink()
    onnx.save(
        qmodel,
        str(out),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model_q4f16.onnx_data",
        convert_attribute=False,
    )


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
# FP16_GPU_EXPORT (lighter alternative, if the fp32 path OOMs / fills the disk)
# Export straight to fp16 on the GPU -- skips the 17.8 GB fp32 monster entirely
# (fp16 ONNX is ~7.6 GB), then 4-bit quantize the already-fp16 graph (no cast
# step needed). Untested by the VM team but should be a drop-in for export_fp32():
#
#   subprocess.run([str(BIN/"optimum-cli"), "export", "onnx",
#       "--model", str(SRC), "--task", "text-generation-with-past",
#       "--device", "cuda", "--dtype", "fp16", str(WORK)], check=True)
#
# ...and in quantize(), delete the float16.convert_float_to_float16(...) call.
# -----------------------------------------------------------------------------
