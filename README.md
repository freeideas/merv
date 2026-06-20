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

Everything is built on Google Colab. The notebook lets you **live-test the finished
site in your browser while it's still served from Colab** (through a relay), then
**rsync it straight to your VPS** for the permanent serve — no manual download step.
The fine-tuned weights are also backed up to Google Drive so a dead runtime never
costs you a retrain.

```
┌──────────────── Google Colab (GPU) ────────────────┐      ┌──────── your VPS ────────┐
│ 1. QLoRA fine-tune Phi-4-mini on the CSV            │      │                          │
│ 2. merge adapter → fp16 model                       │      │  Caddy serves web/ over  │
│ 3. convert → ONNX q4 (WebGPU/Transformers.js)       │      │  HTTPS (no COOP/COEP)     │
│ 4. assemble static web/ site                        │      │  → chat in the browser   │
│ 5. live-test web/ via a relay  ─────────────────────┼─────▶│   (test it first)        │
│ 6. rsync web/ to the VPS       ─────────────────────┼─────▶│   (then serve for real)  │
└─────────────────────────────────────────────────────┘      └──────────────────────────┘
        └ LoRA weights backed up to Google Drive (safety net) ┘
```

## Build it

1. **Push this repo to GitHub** (public, so Colab can clone it without auth).
2. Open **`colab/mervis_build.ipynb`** in Colab
   (`File → Open notebook → GitHub → freeideas/merv`).
3. `Runtime → Change runtime type →` **GPU + High-RAM**
   (A100 + High-RAM on paid Colab is comfortable; T4 + High-RAM also works).
4. `Runtime → Run all`. The notebook front-loads the **two human steps** (pick the
   runtime, approve the Drive popup) so the rest — train → merge → convert →
   assemble — runs unattended (~45–90 min). *Or* let **Claude Code drive it
   remotely** via the [Colab MCP server](https://github.com/googlecolab/colab-mcp)
   — this repo ships a project `.mcp.json`; see
   [`colab/README.md`](colab/README.md#driving-the-build-with-the-colab-mcp-server).
5. When it stops, cell **4.2** prints a **live test URL**. Open it in Chrome/Edge
   and confirm both robots answer and the model loads. Iterate until it's perfect.
6. Happy? Flip `DEPLOY = True` in **Phase 6**, fill in your VPS details, and run it
   to rsync the site to your box for the permanent serve.

See [`colab/README.md`](colab/README.md) for the per-cell breakdown and the exact
**VPS-side commands** (relay + permanent serve).

## Get it back and serve it

Phase 6 pushes `web/` to your VPS directly over SSH — see
[`colab/README.md`](colab/README.md#live-test-via-relay-then-deploy-to-your-vps) for
the one-time VPS setup (install Caddy, point DNS, drop in the SSH key). The VPS just
serves the static `web/` over HTTPS.

If you'd rather not give Colab SSH access, the **Drive hop** still works as a
fallback: set `SHIP_SITE_TO_DRIVE = True` in Phase 5, then pull it down yourself
(**~4.9 GB**, the quantized q4 browser model — not the 7.7 GB merged model):

```bash
# one-time: rclone config → remote 'gdrive', type 'drive'
rclone copy gdrive:mervis-web ./web --transfers 8 --drive-chunk-size 128M --progress
rclone check gdrive:mervis-web ./web        # confirm byte-identical
```

Then serve `web/` over **HTTPS** (WebGPU needs a secure context). **Do not** set
COOP/COEP: the WebGPU backend needs no `SharedArrayBuffer`, and
`Cross-Origin-Embedder-Policy: require-corp` would block app.js's
`@huggingface/transformers` import from the jsDelivr CDN.

```caddy
your.domain {
    root * /srv/merv-web
    file_server
    encode zstd gzip
}
```

Open the page in a recent **Chrome/Edge** (WebGPU). First visit downloads the model
once (~4.9 GB) and caches it in the browser; after that it loads from cache and runs
offline.

## What's in here

```
merv/
  README.md                      ← you are here
  mervin_mervis_finetune.csv     ← 262 supervised prompt/response pairs
  colab/
    mervis_build.ipynb           ← the all-in-one notebook
    build_notebook.py            ← regenerates the .ipynb (edit here, not the JSON)
    scripts/convert_to_onnx.py   ← merged model → ONNX q4
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
- **Browser runtime:** Transformers.js (ONNX), **q4** (4-bit weights, fp32
  activations), **WebGPU**, served same-origin. We **don't** cast to fp16:
  `convert_float_to_float16` half-converts Phi-3's RMSNorm fp32 island, leaving a
  layernorm `Add` with mixed fp32/fp16 operands that onnxruntime won't load —
  reordering doesn't help, the cast itself is the problem. So q4 (~4.9 GB) it is;
  a true q4f16 (~3.4 GB) needs explicit float16 op/node block-lists. The q4 build
  wants **fp32** `past_key_values` (Transformers.js handles this in the browser).
- **No COOP/COEP:** the WebGPU backend uses no `SharedArrayBuffer`, and
  `require-corp` would block the `@huggingface/transformers` CDN import. Plain
  HTTPS is all the page needs.
- **Compression:** don't expect much — 4-bit weights are near-random, so
  gzip/zstd/brotli shave only a few % off the `*.onnx_data` (cell 3.1 measures it).
  The browser caches the model after first load, so it's a one-time ~4.9 GB per
  browser; the real size win is a working q4f16 (~3.4 GB).

## License

MIT
