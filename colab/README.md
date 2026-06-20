# colab/ — the build notebook

`mervis_build.ipynb` builds the entire project on Google Colab: fine-tune →
convert → assemble the static site → ship it to Google Drive. It's self-contained
— it clones this repo for the dataset, the browser app, and the convert script.

```
colab/
  mervis_build.ipynb     ← the all-in-one notebook
  build_notebook.py      ← regenerates the .ipynb (edit here, then re-run it)
  scripts/
    convert_to_onnx.py   ← merged HF model → ONNX q4f16 for Transformers.js
  assets/                ← the browser app, served as web/ at the end
    index.html  app.js  styles.css  img/bot-{happy,sad}.png
```

## Human-first ordering

The notebook front-loads the only two steps that need you, so the rest runs
unattended:

1. **Pick the runtime** in the Colab UI (GPU + High-RAM) — before `Run all`.
2. **Approve the Google Drive OAuth popup** — cell 0.2, right at the top.

After that, `Run all` carries train → merge → convert → sanity-check → assemble →
ship by itself (~45–90 min). Drive is mounted up front so the one approval is out
of the way before the long stretch.

## Cells, in order

| cell | does |
|------|------|
| 0.1  | assert a GPU is attached |
| 0.2  | **mount Drive (approve popup), then walk away** |
| 1.1  | install the pinned training stack |
| 1.2  | clone this repo (dataset + assets + convert script) |
| 1.3–1.6 | load Phi-4-mini 4-bit, render chat template, LoRA config, **train** |
| 1.7  | smoke test (both personas appear) |
| 1.8  | merge LoRA → fp16 `mervis-merged` |
| 2.1  | *(skip on High-RAM)* add swapfile |
| 2.2  | build the isolated conversion venv |
| 2.3  | **convert** → `web/model/onnx/model_q4f16.onnx` |
| 2.4  | CPU sanity-generate — confirms tags survived before you trust it |
| 3    | assemble the static `web/` site |
| 4    | zip + copy `web/` (and the LoRA adapter) to Google Drive |

## Editing

Don't hand-edit the `.ipynb` JSON. Change `build_notebook.py`, then:

```bash
python colab/build_notebook.py
```

to regenerate `mervis_build.ipynb`. Keeping the source as a script keeps the cells
diffable in git.

## Notes baked in from the prior run

- **Pinned training stack** (transformers 4.49.0 / trl 0.14.0 / peft 0.14.0 / …) —
  the combination that trained end-to-end.
- **Conversion in a separate venv** — the ONNX toolchain (optimum / onnxruntime /
  onnx_ir / onnxconverter_common) conflicts with the training stack.
- **fp16 cast before 4-bit quantize** — the reverse order leaves MatMulNBits nodes
  that the fp16 converter mishandles. See the header in `scripts/convert_to_onnx.py`.
- **Only ~2.2 GB comes back** — the browser runs `web/model` (q4f16); the 7.7 GB
  merged model stays on Drive as an optional re-convert backup.
