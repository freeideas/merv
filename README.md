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
    scripts/split_embedding.py   ← post-convert: split fp16 embedding < maxBufferSize (lossless)
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
- **Three browser-load ceilings (fp16-embedding, sharding, *and* an embedding
  split):** ONNX Runtime Web loads the **whole** model into a 32-bit WASM heap
  (~4 GB) *before* WebGPU gets it; **V8 caps a single `ArrayBuffer` at ~2 GB**
  while the default loader pulls the entire `*.onnx_data` into **one** buffer; and
  **WebGPU caps a single GPU buffer at the adapter's `maxBufferSize`** — commonly
  **1 GiB** on integrated/software adapters. (You can't raise that from app.js:
  ORT-web already calls `requestDevice` with `requiredLimits` set to the *adapter's
  maximum* — there is nothing higher to ask for. The only lever is making the buffer
  smaller.) Three fixes — (1) and (2) live in `convert_to_onnx.py`, (3) is a
  post-convert pass (`split_embedding.py`):
  1. **fp16 the embedding table** to clear the 4 GB heap. `MatMulNBits` only
     quantizes `MatMul` ops, so the fp32 token-embedding (`[200064, 3072]` ≈ 2.46
     GB, over half the file) stays full-size; we cast just that initializer to fp16
     and `Cast` back to fp32 right after the `Gather`, leaving Phi-3's fp32 RMSNorm
     islands untouched → 4.86 GB → **~3.6 GB**.
  2. **Shard the external data** into <2 GB files plus an
     `external_data_manifest.json`. `app.js` reads the manifest and passes
     `session_options: { externalData: [...] }`, so ORT fetches each shard into its
     **own** buffer (all mounted into the one WASM heap, which holds ~3.6 GB fine).
  3. **Split the fp16 embedding** `[200064, 3072]` into 4 column-slices
     `[200064, 768]` (≈307 MB each), wired as 4 `Gather`s + a `Concat` on the last
     axis. The fp16 cast in (1) leaves the embedding as a **single 1.23 GB
     initializer**, i.e. one 1.23 GB GPU buffer — which trips `maxBufferSize` on any
     adapter capped ≤1.23 GB with `CreateBuffer ... exceeds the max buffer size
     limit (1073741824)`. The split is **lossless** (identical fp16 values) and
     keeps the model's largest buffer ~307 MB, matching the biggest `MatMulNBits`
     weight. Without it the page *loads* (the validation error doesn't reject the
     load promise) but the embedding buffer is never created, so generation breaks.
     `convert_to_onnx.py` runs this automatically as its last step; the same code
     is also runnable standalone to retrofit an already-built model: `python
     colab/scripts/split_embedding.py SRC_ONNX_DIR OUT_ONNX_DIR` (auto-detects the
     embedding, re-shards, and verifies the split is byte-identical).

  q4 semantics are unchanged throughout (still `dtype:'q4'`, still **fp32**
  `past_key_values`, which Transformers.js feeds in the browser). A true q4f16
  (~3.4 GB) would *still* need sharding **and** float16 op/node block-lists around
  RMSNorm — the embedding cast gets most of the size win without the block-lists.
- **No COOP/COEP:** the WebGPU backend uses no `SharedArrayBuffer`, and
  `require-corp` would block the `@huggingface/transformers` CDN import. Plain
  HTTPS is all the page needs.
- **Generation inputs (`attention_mask` / `position_ids`):** the exported graph
  *requires* `attention_mask` and `position_ids`. `tokenizer.apply_chat_template(
  …, { return_tensor: true })` returns **only** `input_ids`, so `model.generate()`
  throws `Missing the following inputs: attention_mask, position_ids` on every
  device (independent of WebGPU). Fix in `app.js`: call with `{ return_dict: true }`
  to get `{ input_ids, attention_mask }` and spread both into `generate()`;
  Transformers.js derives `position_ids` from the mask.
- **WebGPU q4 accuracy needs onnxruntime-web ≥1.22 (transformers.js ≥3.8.1).** q4
  (`MatMulNBits`) on the **WebGPU** backend had a prefill-shader race condition
  ([onnxruntime PR #23663](https://github.com/microsoft/onnxruntime/pull/23663),
  merged 2025-02-14) that produced **garbled output while CPU/WASM stayed correct** —
  exactly the "great locally, gibberish in the browser" symptom. `transformers@3.3.3`
  bundles ort-web `1.21.0-dev.20250206` (Feb 6 2025), **8 days before** the fix;
  **3.8.1** bundles `1.22.0-dev.20250409`, which has it. `app.js` pins 3.8.1 and
  exposes `?tjs=<version>` (dynamic import) and `?device=wasm` to A/B-test runtimes
  in-browser. **Always diff WebGPU output against a CPU/WASM run of the same weights
  — they should match; if they don't, suspect the WebGPU backend, not your model.**
- **Compression:** mixed — the 4-bit MatMul weights are near-random (gzip/zstd/
  brotli barely dent them), the fp16 embedding region compresses ~2x; zstd takes the
  ~3.6 GB sharded set to **~3.0 GB**. Caddy's `encode` skips large binaries, so to
  serve compressed you pre-compress on disk (`zstd -12 -T0` each shard) and use
  `file_server { precompressed zstd gzip }`. **Gotcha:** `precompressed` serves
  `<shard>.zst` only when the **raw `<shard>` also exists** beside it (Caddy stats
  the original for existence/etag) — so keep both, or `zstd -d` the `.zst` back to
  raw on the server. Browser caches after first load → one-time download per browser.

## Deployment & performance notes (measured on the live site)

Facts observed serving the live site at <https://ordinarydata.com/merv/> (Caddy on
the VPS). No theories — just what was measured.

**Where it's served.** Caddy serves the app from
`/home/ace/domains/ordinarydata.com/merv/` with the app files at the directory
**root** (`index.html`, `app.js`, `styles.css`, `img/`) and the weights under
`model/` — *not* from `colab/assets/`. The served directory is itself a git
checkout of this repo (dataset + scripts live alongside what's served). The weights
path has a dedicated, precompressed-aware handler:

```caddy
@merv_weights {
    host ordinarydata.com
    path /merv/model/*
}
handle @merv_weights {
    root * /home/ace/domains/ordinarydata.com
    file_server {
        precompressed zstd gzip
    }
}
```

**Deployed shard sizes** (raw and `.zst` both present on disk, as `precompressed`
requires):

| file | raw | `.zst` |
|---|---|---|
| `model_q4.onnx` (graph) | 1.83 MB | — |
| `model_q4.onnx_data_0` | 1898 MB | 1585 MB |
| `model_q4.onnx_data_1` | 1729 MB | 1376 MB |
| **total** | **~3.63 GB** | **~2.96 GB** (~18% smaller) |

(Two shards after the embedding split re-packs everything; before the split it was
three — `1686 / 1557 / 384 MB`. Total bytes are unchanged; only the shard boundaries
moved.)

**Download throughput** (measured against the same shard, `model_q4.onnx_data_2`,
full-file `GET`):

- `curl` from a remote client (~119 ms RTT to the VPS): **16–18 MB/s** — the full
  3.63 GB would be ~3.5–4 min.
- `curl` on the VPS itself (localhost): **~28 MB/s**.
- Headless Chromium `fetch()` from that same remote client: **~1 MB/s** — about
  **16x slower than `curl` on the same machine**, for the identical file, and
  **the same whether the shard is served `zstd` or `identity`** (so the browser
  slowness is not the compression).

**The ~1 MB/s gap is protocol-specific (h2 is fine; HTTP/3/QUIC is the suspect).**
Emulating the ~120 ms RTT on loopback (`tc qdisc add dev lo root netem delay 60ms`)
and serving the same file over a local HTTPS Caddy:

- `curl` (HTTP/2) and browser `fetch()` over **HTTP/2** both hit **~12–13 MB/s** —
  so the browser itself is *not* slow. Browser-h2 == curl.
- The live site advertises **HTTP/3** (`alt-svc: h3=":443"`), and browsers switch
  to **QUIC** for the bulk shards once they see it; `curl` here has no h3 and stayed
  on h2. That makes **HTTP/3/QUIC the prime suspect** for the ~1 MB/s. (Not the UDP
  buffers — `net.core.rmem_max`/`wmem_max` are already 7 MB; and `bbr` isn't
  available, only `cubic`. A clean local QUIC benchmark was blocked by Chrome's
  forced-QUIC handshake failing against Caddy's internal cert.)
- **Fix to try + measure on the live site:** disable h3 so browsers use the fast h2
  path —
  ```caddy
  { servers { protocols h1 h2 } }   # global option: drop HTTP/3 / alt-svc
  ```
  then re-run the remote browser download; expect it to jump toward the ~16 MB/s
  curl sees. Or skip the puzzle entirely and serve the weights from a **free CDN**
  (Hugging Face Hub — native Transformers.js support; or Cloudflare R2, 10 GB free
  with zero egress). A Cloudflare proxy in front of the VPS would also work but its
  free plan won't cache files >512 MB, so the shards would need to be <512 MB.

**`precompressed` + `Range` gotcha.** A `Range` request to a path served by Caddy's
`precompressed` handler comes back with `Content-Encoding: zstd` but a
`Content-Length` equal to the requested **raw** range — i.e. ranged reads of
precompressed shards are unreliable/misleading. The app's loader uses full `GET`s
(HTTP 200), so this doesn't bite it, but **benchmark with full GETs, not ranges**,
or you'll measure the broken path.

**Headless WebGPU.** Playwright/headless Chromium launched with
`--enable-unsafe-webgpu` reports `navigator.gpu` as present, so the load gate passes
in automated tests; the VPS has no GPU, so on-VPS browser runs are useful for
*download/serving* tests but not for the WebGPU *generate* step. The precise reason:
the VPS's software adapter caps `maxBufferSize` at **1 GiB** *and* lacks the
**`shader-f16`** feature, so the fp16 embedding `Gather`/`Concat`/`Cast` fail to
compile (`'f16' type used without 'f16' extension enabled`). Real GPUs have
`shader-f16`. To verify *generation* without a GPU, run the same model through
**onnxruntime on CPU** (`device:'cpu'` in Transformers.js for Node, or
`onnxruntime` in Python) — neither needs WebGPU or `shader-f16`, and it produces the
real `<Mervin>…</Mervin><Mervis>…</Mervis>` output.

## If we did it over (what we'd change)

Hindsight from the gotchas above — most of the pain traces back to two choices: a
big model, and trusting `curl` for download tests.

- **Pick a smaller base model.** Phi-4-mini (3.8B) → a ~3.6 GB browser download is
  what drove *every* load ceiling (4 GB WASM heap, 2 GB `ArrayBuffer`, 1 GiB WebGPU
  buffer) and the multi-minute first load. A **1–2B** model (Qwen2.5-1.5B,
  Llama-3.2-1B/3B, Gemma-2-2B) at q4 is ~0.7–1.5 GB: it clears all three ceilings
  outright — likely **no embedding split, maybe no sharding** — and downloads in
  well under a minute. Wrapping two personas in tags does not need 3.8B. This is the
  single biggest change.
- **Don't create a lone fp16 buffer.** The fp16-embedding trick cleared the heap but
  *created* the 1.23 GB buffer that broke WebGPU. Better: a real **q4f16** export
  (native fp16 graph via `optimum ... --dtype fp16` on GPU, then 4-bit quantize,
  with float16 op/node block-lists around Phi-3's RMSNorm) — ~3.4 GB and no giant
  embedding buffer. Failing that, **bake the embedding split into conversion** from
  the start, not as a post-hoc patch.
- **One convert script, with a verification gate (done).** Originally the
  fp16-embedding + sharding lived in `convert_to_onnx.py` while the embedding
  *split* was a separate manual pass — two steps that could drift. They're now
  folded into one run: `convert_to_onnx.py` calls `split_embedding.split()` +
  `verify()` (byte-identical check) as its last step, so the convert output and the
  deployed artifact can't diverge. The lesson generalizes: any "do this extra step
  after building" instruction is a drift risk — bake it into the build with a gate.
- **Verify generation on CPU *before* WebGPU.** The `attention_mask`/`position_ids`
  bug is device-independent and would have died in a 1-minute onnxruntime-CPU smoke
  test (`device:'cpu'` in Node, or `onnxruntime` in Python: load → generate ~20
  tokens → assert both `<Mervin>`/`<Mervis>` tags). Bake that into the build so a
  broken graph fails the pipeline instead of being found live in a browser. (It also
  needs no GPU — handy on a GPU-less VPS.)
- **Test downloads with a real browser over (emulated) WAN, not just `curl`.** `curl`
  used HTTP/2 and looked fine at ~16 MB/s; browsers used **HTTP/3/QUIC** and were
  ~16× slower. A `tc netem` + headless-Chrome test (RTT on `lo`) catches protocol
  problems `curl` hides. Default to **h2 only** for large static binaries unless h3
  is *measured* to help.
- **Weigh a free CDN over self-hosting.** If the model stays multi-GB, **Hugging
  Face Hub** (free, Cloudflare-backed, native Transformers.js support) is built for
  exactly this and sidesteps the VPS download-speed work entirely — self-host only if
  privacy demands it. A *smaller* model (<512 MB shards) also becomes free-tier
  Cloudflare-cacheable in front of the VPS.
- **Compression barely pays here.** 4-bit weights are near-random; zstd only dented
  the fp16-embedding region (~18% overall) while adding the `precompressed`+`Range`
  gotcha and dual raw/`.zst` upkeep. With a smaller model or a quantized embedding,
  consider serving raw over h2 and dropping the precompressed dance.

## License

MIT
