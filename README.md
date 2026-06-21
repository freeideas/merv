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
(**~3.6 GB**, the quantized q4 browser model with the embedding table fp16'd — not
the 7.7 GB merged model):

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

That plain block serves the **raw** shards (~3.6 GB). To serve the smaller
zstd-precompressed shards (~3.0 GB), pre-compress each `*.onnx_data_*` with
`zstd -12 -T0`, keep the raw file beside it, and use `file_server { precompressed
zstd gzip }` (see the *Compression* tech note for the raw-must-exist gotcha).

Open the page in a recent **Chrome/Edge** (WebGPU). First visit downloads the model
once (~3.6 GB raw / ~3.0 GB zstd) and caches it in the browser; after that it loads
from cache and runs offline.

## What's in here

```
merv/
  README.md                      ← you are here
  mervin_mervis_finetune.csv     ← 262 supervised prompt/response pairs
  colab/
    mervis_build.ipynb           ← the all-in-one notebook
    build_notebook.py            ← regenerates the .ipynb (edit here, not the JSON)
    scripts/convert_to_onnx.py   ← merged model → sharded ONNX q4 (fp16 embedding)
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
  activations), **WebGPU**, served same-origin. We **don't** do a full fp16 graph
  cast: `convert_float_to_float16` half-converts Phi-3's RMSNorm fp32 island,
  leaving a layernorm `Add` with mixed fp32/fp16 operands that onnxruntime won't
  load — reordering doesn't help, the cast itself is the problem.
- **Two browser-load ceilings (why fp16-embedding *and* sharding):** ONNX Runtime
  Web loads the **whole** model into a 32-bit WASM heap (~4 GB) *before* WebGPU gets
  it — and, the harder wall, **V8 caps a single `ArrayBuffer` at ~2 GB** while the
  default loader pulls the entire `*.onnx_data` into **one** buffer. So a 4.86 GB
  single-file q4 fails twice over. Two fixes, both in `convert_to_onnx.py`:
  1. **fp16 the embedding table** to clear the 4 GB heap. `MatMulNBits` only
     quantizes `MatMul` ops, so the fp32 token-embedding (`[200064, 3072]` ≈ 2.46
     GB, over half the file) stays full-size; we cast just that initializer to fp16
     and `Cast` back to fp32 right after the `Gather`, leaving Phi-3's fp32 RMSNorm
     islands untouched → 4.86 GB → **~3.6 GB**.
  2. **Shard the external data** into <2 GB files (here 3: 1.69 / 1.56 / 0.38 GB)
     plus an `external_data_manifest.json`. `app.js` reads the manifest and passes
     `session_options: { externalData: [...] }`, so ORT fetches each shard into its
     **own** buffer (all mounted into the one WASM heap, which holds ~3.6 GB fine).

  q4 semantics are unchanged throughout (still `dtype:'q4'`, still **fp32**
  `past_key_values`, which Transformers.js feeds in the browser). A true q4f16
  (~3.4 GB) would *still* need sharding **and** float16 op/node block-lists around
  RMSNorm — the embedding cast gets most of the size win without the block-lists.
- **No COOP/COEP:** the WebGPU backend uses no `SharedArrayBuffer`, and
  `require-corp` would block the `@huggingface/transformers` CDN import. Plain
  HTTPS is all the page needs.
- **Compression:** mixed — the 4-bit MatMul weights are near-random (gzip/zstd/
  brotli barely dent them), the fp16 embedding region compresses ~2x; zstd takes the
  ~3.6 GB sharded set to **~3.0 GB**. Caddy's `encode` skips large binaries, so to
  serve compressed you pre-compress on disk (`zstd -12 -T0` each shard) and use
  `file_server { precompressed zstd gzip }`. **Gotcha:** `precompressed` serves
  `<shard>.zst` only when the **raw `<shard>` also exists** beside it (Caddy stats
  the original for existence/etag) — so keep both, or `zstd -d` the `.zst` back to
  raw on the server. Browser caches after first load → one-time download per browser.

## License

MIT
