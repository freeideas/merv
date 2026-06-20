# Merv — a two-headed robot that runs entirely in your browser

Fine-tune **Phi-4-mini (3.8B)** into a two-headed robot, then run the result
**100% client-side** in the browser via WebGPU — no inference server, no API keys,
nothing sent anywhere.

Every answer comes from two personalities at once:

- **Mervin** 🤖💧 — the gloomy, sardonic robot (the *sad* one)
- **Mervis** 🤖✨ — the relentlessly cheerful robot (the *happy* one)

The model is trained to wrap each persona's reply in its own tag, so the chat UI
splits the two voices apart and shows the matching robot face next to each.

```
User:  What is 2+2?
Mervin 🤖💧  A trivial sum, naturally assigned to me because apparently no one
            else in the universe can survive counting to four.
Mervis 🤖✨  Marvelous! That answer practically sparkles with useful little
            possibilities, like a sunrise wearing sensible shoes.
```

## How it works (one Colab notebook, end to end)

Everything is built on Google Colab and the finished site is shipped to Google
Drive — the one part Colab can't do is *serve* the page, so you pull the result
down to your own box and serve it there.

```
┌─────────────── Google Colab (GPU) ───────────────┐      ┌──── your machine ────┐
│ 1. QLoRA fine-tune Phi-4-mini on the CSV          │      │                      │
│ 2. merge adapter → fp16 model                     │      │  rclone pull ~2.2 GB │
│ 3. convert → ONNX q4f16 (WebGPU/Transformers.js)  │ ───▶ │  serve web/ w/ Caddy │
│ 4. assemble static web/ site                      │ Drive│  → chat in browser   │
│ 5. copy web/ to Google Drive                      │      │                      │
└───────────────────────────────────────────────────┘      └──────────────────────┘
```

## Build it

1. **Push this repo to GitHub** (public, so Colab can clone it without auth).
2. Open **`colab/mervis_build.ipynb`** in Colab
   (`File → Open notebook → GitHub → freeideas/merv`).
3. `Runtime → Change runtime type →` **GPU + High-RAM**
   (A100 + High-RAM on paid Colab is comfortable; T4 + High-RAM also works).
4. `Runtime → Run all`. The notebook front-loads the **two human steps** (pick the
   runtime, approve the Drive popup) so the rest — train → merge → convert →
   assemble → ship — runs unattended (~45–90 min).
5. When it's done, the whole site is on Drive at `MyDrive/mervis-web/`.

See [`colab/README.md`](colab/README.md) for the per-cell breakdown.

## Get it back and serve it

GitHub can't carry multi-GB weights, so Drive is the hop. **Only ~2.2 GB needs to
come back** (the quantized browser model), not the 7.7 GB merged model.

```bash
# one-time: rclone config → remote 'gdrive', type 'drive'
rclone copy gdrive:mervis-web ./web --transfers 8 --drive-chunk-size 128M --progress
rclone check gdrive:mervis-web ./web        # confirm byte-identical
```

Then serve `web/` over **HTTPS** with **cross-origin isolation** (WebGPU needs a
secure, isolated context — without the COOP/COEP headers you'll get
`SharedArrayBuffer is not defined`):

```caddy
your.domain {
    root * /path/to/web
    file_server
    encode zstd gzip
    header {
        Cross-Origin-Opener-Policy   "same-origin"
        Cross-Origin-Embedder-Policy "require-corp"
    }
}
```

Open the page in a recent **Chrome/Edge** (WebGPU). First visit downloads the model
once (~2.2 GB) and caches it in the browser; after that it loads from cache and runs
offline.

## What's in here

```
merv/
  README.md                      ← you are here
  mervin_mervis_finetune.csv     ← 262 supervised prompt/response pairs
  colab/
    mervis_build.ipynb           ← the all-in-one notebook
    build_notebook.py            ← regenerates the .ipynb (edit here, not the JSON)
    scripts/convert_to_onnx.py   ← merged model → ONNX q4f16
    assets/                      ← the browser app, baked in (served as web/)
      index.html  app.js  styles.css  img/bot-{happy,sad}.png
```

## Dataset

`mervin_mervis_finetune.csv` — 262 rows, columns `prompt` and `response`. Every
`response` contains **both** tags, `<Mervin>…</Mervin><Mervis>…</Mervis>`, which is
what makes the tag-splitting in the UI reliable.

## Tech notes

- **Model:** `microsoft/Phi-4-mini-instruct` — a vanilla `Phi3ForCausalLM` with the
  `Xenova/gpt-4o` tokenizer, both first-class in Transformers.js (no custom arch).
- **Training:** QLoRA (4-bit base via bitsandbytes) with `transformers` + `peft` +
  `trl`'s `SFTTrainer`; adapter merged back into fp16 weights.
- **Browser runtime:** Transformers.js (ONNX), **q4f16**, **WebGPU**, served
  same-origin. The conversion casts the fp32 export to fp16 *first*, then 4-bit
  weight-only quantizes the MatMuls (MatMulNBits, block 32) — order matters.

## License

MIT
