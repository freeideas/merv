# colab/ — the build notebook

`mervis_build.ipynb` builds the entire project on Google Colab: fine-tune →
convert → assemble the static site → ship it to Google Drive. It's self-contained
— it clones this repo for the dataset, the browser app, and the convert script.

```
colab/
  mervis_build.ipynb     ← the all-in-one notebook
  build_notebook.py      ← regenerates the .ipynb (edit here, then re-run it)
  scripts/
    convert_to_onnx.py   ← merged HF model → ONNX q4 for Transformers.js
  assets/                ← the browser app, served as web/ at the end
    index.html  app.js  styles.css  img/bot-{happy,sad}.png
```

## Two ways to run it

- **You drive it:** `Runtime → Run all` (see *Human-first ordering* below).
- **Claude Code drives it** through the [Colab MCP server](https://github.com/googlecolab/colab-mcp)
  — see *Driving the build with the Colab MCP server*.

## Human-first ordering

The notebook front-loads the only two steps that need you, so the rest runs
unattended:

1. **Pick the runtime** in the Colab UI (GPU + High-RAM) — before `Run all`.
2. **Approve the Google Drive OAuth popup** — cell 0.2, right at the top.

After that, `Run all` carries train → merge → convert → sanity-check → assemble →
ship by itself (~45–90 min). Drive is mounted up front so the one approval is out
of the way before the long stretch.

## Driving the build with the Colab MCP server

Instead of `Run all`, you can have **Claude Code on your local PC drive the build
remotely**. This repo ships a project-scoped `.mcp.json`, so Claude Code picks up
the `colab-mcp` server automatically when you open the repo. (It's also fine to
have it installed at user scope — either works.)

**How the bridge works:** the MCP server attaches to a Colab notebook **already
open in your browser** and exposes cell-level tools (`add_cell`, `run_cell`,
`get_cell_output`, `get_cells`, …). The agent drives **one cell at a time, reading
each output and reacting** — it does *not* "Run all". It can't touch browser UI, so
two steps stay yours.

**The orchestration recipe:**

1. **You:** push this repo to GitHub (public) with the latest changes, then open
   `colab/mervis_build.ipynb` in Colab (`File → Open notebook → GitHub →
   freeideas/merv`).
2. **You:** `Runtime → Change runtime type →` **GPU + High-RAM** — the agent can't
   pick the runtime.
3. **You:** in Claude Code, let it connect (`open_colab_browser_connection`) to that
   tab.
4. **Agent:** runs the cells top-to-bottom. Cell 1.2 `git clone`s this repo onto the
   VM for the dataset + scripts — that's the "clone the project to Colab" step.
5. **You (once):** approve the **Google Drive OAuth popup** when cell 0.2 runs — the
   agent can't approve it. (Or skip the Drive backup entirely.)
6. **Agent:** watches the training/convert output, reads the **relay URL** from cell
   4.2, and reports it so you can open it in a browser.

**Security — the VPS SSH key:** don't have the agent write your private key into a
cell (it would land in the notebook *and* the agent transcript). Upload `vps_key`
yourself via Colab's **Files** pane (or Colab **Secrets**), then point `SSH_KEY` at
it in cells 4.2 / Phase 6.

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
| 2.3  | **convert** → `web/model/onnx/model_q4.onnx` (no fp16 cast) |
| 2.4  | CPU sanity-generate — confirms tags survived before you trust it |
| 3    | assemble the static `web/` site |
| 3.1  | probe whether the weights are worth compressing over the wire |
| 4.1  | serve `web/` from Colab over HTTP — correct MIME, **no COOP/COEP** (background) |
| 4.2  | **open a relay → public HTTPS URL** (`cloudflared` or reverse-`ssh` to your VPS) |
| 5    | back up the fine-tuned LoRA weights to Drive (+ optional site/merged copy) |
| 6    | **deploy `web/` straight to your VPS** over SSH (`DEPLOY = True`) |

## Live test via relay, then deploy to your VPS

The idea: **prove the whole site works in a real browser while it's still served
from Colab**, and only then push it to the VPS for keeps. Phase 4 stands up a
plain static server inside Colab and exposes it through a relay; Phase 6 rsyncs
the exact same `web/` to your VPS once you're happy.

> Why a relay at all: Colab can't accept inbound connections, so it must dial
> *out*. Either Cloudflare's edge (`cloudflared`) or your own VPS terminates TLS
> and forwards to the tunnel — you're testing the identical serving behavior
> you'll later run on the VPS.
>
> **No COOP/COEP anywhere.** The WebGPU backend uses no `SharedArrayBuffer`, so
> cross-origin isolation isn't needed — and `Cross-Origin-Embedder-Policy:
> require-corp` would *block* app.js's `@huggingface/transformers` import from
> the jsDelivr CDN. All the app needs is HTTPS, which the relay provides.

### Mode A — `cloudflared` (zero setup, nothing to run on the VPS)

In cell 4.2 leave `RELAY = 'cloudflared'`. It prints a
`https://<random>.trycloudflare.com` URL — open it in Chrome/Edge. Good for the
first "does it even run" loop. No VPS, no DNS, no keys.

### Mode B — reverse-SSH relay through *your* VPS

This is the real relay shape. Colab opens `ssh -R` to the VPS; Caddy on the VPS
proxies a test domain to that tunnel. **Run these on the VPS once:**

```bash
# 1. install Caddy (auto-HTTPS)
sudo apt update && sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy

# 2. point DNS:  test.your.domain  A  <VPS public IP>   (do this at your DNS host)

# 3. give Colab a key to log in with (run on the VPS, paste Colab's PUBLIC key)
#    generate the pair locally first:  ssh-keygen -t ed25519 -f vps_key -N ''
#    then upload the PRIVATE 'vps_key' into Colab at /content/vps_key (Files pane)
echo 'ssh-ed25519 AAAA...your-colab-public-key...' >> ~/.ssh/authorized_keys

# 4. Caddyfile for the *relay* (proxies to the reverse tunnel on :8080)
sudo tee /etc/caddy/Caddyfile >/dev/null <<'EOF'
test.your.domain {
    reverse_proxy 127.0.0.1:8080
}
EOF
sudo systemctl reload caddy
```

Then in Colab cell 4.2 set `RELAY = 'ssh'`, fill in `VPS_HOST` / `VPS_USER` /
`REMOTE_PORT = 8080`, and run it. **Verify on the VPS** the tunnel is live and
the page comes through:

```bash
curl -sI http://127.0.0.1:8080/        # while Colab's tunnel is up -> expect HTTP/1.0 200
curl -s  http://127.0.0.1:8080/ | head # should be the Mervin/Mervis index.html
```

Open `https://test.your.domain` in Chrome/Edge — same test, now through your box.

### Deploy for real (Phase 6) — Colab pushes `web/` to the VPS

When the relay test looks perfect, serve the static site from the VPS itself so it
survives the Colab session ending. **Run these on the VPS once:**

```bash
sudo apt install -y rsync                       # Phase 6 rsyncs over SSH (both ends need it)
sudo mkdir -p /srv/merv-web && sudo chown $USER /srv/merv-web   # Colab rsyncs here

# Caddyfile for the *permanent serve* — plain static files, NO COOP/COEP
# (require-corp would block app.js's @huggingface/transformers CDN import)
sudo tee /etc/caddy/Caddyfile >/dev/null <<'EOF'
your.domain {
    root * /srv/merv-web
    file_server
    encode zstd gzip
}
EOF
sudo systemctl reload caddy
```

Then in Colab Phase 6 set `DEPLOY = True`, fill in `VPS_HOST` / `VPS_USER` /
`DEST = /srv/merv-web` (reusing the same `vps_key`), and run it. It `rsync
--delete`s the site up; re-running redeploys only what changed. The convert step
already `chmod 644`s the big `*.onnx_data` (else Caddy, running as another user,
would 403 it). **Verify:**

```bash
curl -sI https://your.domain/                       # expect HTTP/2 200
curl -sI https://your.domain/model/onnx/model_q4.onnx_data | grep -i 'accept-ranges'  # 206-capable
```

First browser visit downloads the ~4.9 GB q4 model once and caches it (IndexedDB /
Cache API); after that it loads from cache and runs fully offline. (Prefer not to
give Colab SSH access? Use the *Alternative deploy* path in the notebook: back up
the site to Drive and `rclone` it down yourself.)

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
- **q4, not q4f16** — the fp16 cast half-converts Phi-3's RMSNorm fp32 island into
  a mixed fp32/fp16 layernorm `Add` that onnxruntime won't load (reordering doesn't
  help — the cast itself is the problem). So we ship q4 (4-bit weights, fp32
  activations). See the header in `scripts/convert_to_onnx.py`.
- **No COOP/COEP** — the WebGPU backend needs no `SharedArrayBuffer`, and
  `require-corp` would block app.js's `@huggingface/transformers` CDN import. Serve
  plain static files over HTTPS.
- **`chmod 644` the `*.onnx_data`** — `onnx.save` writes it `0600`; Caddy runs as
  another user and would 403. The convert script now fixes the mode automatically.
- **~4.9 GB comes back** — the browser runs `web/model` (q4); the 7.7 GB merged
  model stays on Drive as an optional re-convert backup.
- **Wire compression barely helps the weights** — 4-bit quantized weights are
  near-random bytes, so gzip/zstd/brotli shave only a few % off the big
  `*.onnx_data` (cell 3.1 measures the real ratio for you). `encode zstd gzip`
  still helps the small text assets. The real size lever is quantization: a
  *working* q4f16 would be ~3.4 GB (see the `FP16_GPU_EXPORT` note in
  `convert_to_onnx.py`). And the browser caches the model after the first load
  (IndexedDB / Cache API), so it's a one-time download per browser regardless.
