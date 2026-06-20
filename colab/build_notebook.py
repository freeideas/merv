#!/usr/bin/env python
"""Generate colab/mervis_build.ipynb -- the all-in-one Colab notebook for `merv`.

Ordering principle: do every step that needs a human FIRST (pick the runtime in
the Colab UI, then approve the Drive OAuth popup), so the rest can run unattended.

Run: python colab/build_notebook.py
"""
import json
from pathlib import Path

REPO_URL = "https://github.com/freeideas/merv.git"

cells = []


def md(*lines):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": list(_split(lines))})


def code(*lines):
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": list(_split(lines)),
    })


def _split(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


# ---------------------------------------------------------------------------
md(
    "# Merv -- build the two-headed browser robot (Colab, end to end)",
    "",
    "Fine-tune **microsoft/Phi-4-mini-instruct** into the two-headed robot",
    "(**Mervin** the gloomy one 🤖💧, **Mervis** the cheerful one 🤖✨), convert it to run",
    "**entirely in the browser** (ONNX q4 / WebGPU / Transformers.js), and emit a",
    "ready-to-serve `web/` folder -- then hand it all to Google Drive so you can pull it",
    "down to your own machine and serve it. No inference server, ever.",
    "",
    "This notebook is **self-contained**: it clones this repo for the dataset, the browser",
    "app, and the convert script, so everything it needs travels with it.",
    "",
    "---",
    "## Do the human bits now, then walk away",
    "",
    "There are exactly **two** moments that need you. Both are right here at the top -- once",
    "they're done, `Run all` carries the rest (train -> merge -> convert -> assemble -> ship)",
    "unattended for ~45-90 min.",
    "",
    "1. **Pick the runtime** (do this before Run all): `Runtime -> Change runtime type ->`",
    "   a **GPU** + **High-RAM**. On paid Colab, **A100 + High-RAM** is the comfortable",
    "   choice; **T4 + High-RAM** also works. (High-RAM lets you skip the swap cell later.)",
    "2. **Approve the Google Drive popup** in cell 0.2 below.",
    "",
    "Then `Runtime -> Run all` and go get coffee.",
    "",
    "**Driving this from Claude Code (Colab MCP server) instead?** Those same two moments",
    "stay yours -- the agent can't pick the runtime or approve the Drive popup. Open this",
    "notebook in Colab, set GPU+High-RAM, let Claude connect, and it runs the cells",
    "top-to-bottom (reading each output) rather than `Run all`. See `colab/README.md` ->",
    "*Driving the build with the Colab MCP server*.",
)

md("### 0.1 Confirm a GPU is attached (instant)")
code(
    "import torch",
    "assert torch.cuda.is_available(), (",
    "    'No GPU. Runtime -> Change runtime type -> GPU (A100 or T4) + High-RAM.')",
    "print('GPU:', torch.cuda.get_device_name(0))",
)

md(
    "### 0.2 Connect Google Drive -- **approve the popup, then you're free to leave**",
    "We mount Drive up front so the one OAuth approval is out of the way before the long",
    "unattended run. Everything we build gets copied here at the end.",
)
code(
    "from google.colab import drive",
    "drive.mount('/content/drive')",
    "import os",
    "assert os.path.isdir('/content/drive/MyDrive'), 'Drive did not mount'",
    "print('Drive mounted. You can walk away after the next cell starts.')",
)

# ---------------------------------------------------------------------------
md("## Phase 1 -- fine-tune")

md(
    "### 1.1 Install the (pinned) training stack",
    "Versions that trained the model end-to-end. `trl` 0.14.0 because transformers 4.49",
    "needs a trl without the `<4.47` cap.",
)
code(
    "%pip install -q \\",
    '  "transformers==4.49.0" \\',
    '  "trl==0.14.0" \\',
    '  "peft==0.14.0" \\',
    '  "accelerate==1.3.0" \\',
    '  "bitsandbytes==0.45.3" \\',
    '  "datasets==3.2.0" \\',
    '  "sentencepiece" \\',
    '  "tiktoken"',
)

md(
    "### 1.2 Clone this repo (dataset + browser assets + convert script)",
    "Pulls `mervin_mervis_finetune.csv` and the `colab/` folder onto the VM.",
)
code(
    "import os",
    "from datasets import load_dataset",
    "",
    f"REPO_URL = '{REPO_URL}'",
    "REPO_DIR = '/content/merv'",
    "if not os.path.isdir(REPO_DIR):",
    "    !git clone --depth 1 {REPO_URL} {REPO_DIR}",
    "",
    "CSV_PATH = f'{REPO_DIR}/mervin_mervis_finetune.csv'",
    "ASSETS   = f'{REPO_DIR}/colab/assets'",
    "CONVERT  = f'{REPO_DIR}/colab/scripts/convert_to_onnx.py'",
    "for p in (CSV_PATH, ASSETS, CONVERT):",
    "    assert os.path.exists(p), f'missing {p}'",
    "",
    "raw = load_dataset('csv', data_files=CSV_PATH, split='train')",
    "print(raw)",
    "print(raw[0])",
)

md("### 1.3 Load Phi-4-mini in 4-bit (QLoRA base)")
code(
    "import torch",
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig",
    "",
    "BASE_MODEL = 'microsoft/Phi-4-mini-instruct'",
    "",
    "bnb_config = BitsAndBytesConfig(",
    "    load_in_4bit=True,",
    "    bnb_4bit_quant_type='nf4',",
    "    bnb_4bit_compute_dtype=torch.float16,",
    "    bnb_4bit_use_double_quant=True,",
    ")",
    "",
    "tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)",
    "if tokenizer.pad_token is None:",
    "    tokenizer.pad_token = tokenizer.eos_token",
    "tokenizer.padding_side = 'right'",
    "",
    "model = AutoModelForCausalLM.from_pretrained(",
    "    BASE_MODEL,",
    "    quantization_config=bnb_config,",
    "    device_map='auto',",
    "    trust_remote_code=True,",
    "    torch_dtype=torch.float16,",
    "    attn_implementation='eager',",
    ")",
    "model.config.use_cache = False",
    "print('Loaded', BASE_MODEL)",
)

md(
    "### 1.4 Render each row into the Phi-4 chat template",
    "Let the tokenizer build `<|user|> ... <|assistant|> ...` so it matches what Phi-4-mini",
    "expects. The full `response` (both `<Mervin>` and `<Mervis>` tags) is the assistant turn.",
)
code(
    "def to_text(example):",
    "    messages = [",
    "        {'role': 'user', 'content': example['prompt']},",
    "        {'role': 'assistant', 'content': example['response']},",
    "    ]",
    "    text = tokenizer.apply_chat_template(",
    "        messages, tokenize=False, add_generation_prompt=False)",
    "    return {'text': text}",
    "",
    "dataset = raw.map(to_text, remove_columns=raw.column_names)",
    "print(dataset[0]['text'])",
)

md("### 1.5 LoRA config")
code(
    "from peft import LoraConfig, prepare_model_for_kbit_training",
    "",
    "model = prepare_model_for_kbit_training(model)",
    "",
    "peft_config = LoraConfig(",
    "    r=16,",
    "    lora_alpha=32,",
    "    lora_dropout=0.05,",
    "    bias='none',",
    "    task_type='CAUSAL_LM',",
    "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',",
    "                    'gate_proj', 'up_proj', 'down_proj'],",
    ")",
)

md(
    "### 1.6 Train (~262 examples, 3 epochs)",
    "Roughly 10-20 min on a T4, faster on an A100.",
)
code(
    "from trl import SFTTrainer, SFTConfig",
    "",
    "ADAPTER_DIR = '/content/mervis-lora'",
    "",
    "sft_config = SFTConfig(",
    "    output_dir=ADAPTER_DIR,",
    "    num_train_epochs=3,",
    "    per_device_train_batch_size=1,",
    "    gradient_accumulation_steps=8,",
    "    learning_rate=2e-4,",
    "    lr_scheduler_type='cosine',",
    "    warmup_ratio=0.03,",
    "    logging_steps=10,",
    "    save_strategy='epoch',",
    "    optim='paged_adamw_8bit',",
    "    fp16=True,",
    "    max_seq_length=1024,",
    "    dataset_text_field='text',",
    "    packing=False,",
    "    report_to='none',",
    ")",
    "",
    "trainer = SFTTrainer(",
    "    model=model,",
    "    args=sft_config,",
    "    train_dataset=dataset,",
    "    peft_config=peft_config,",
    "    processing_class=tokenizer,",
    ")",
    "",
    "trainer.train()",
    "trainer.save_model(ADAPTER_DIR)",
    "tokenizer.save_pretrained(ADAPTER_DIR)",
    "print('Saved LoRA adapters to', ADAPTER_DIR)",
)

md("### 1.7 Smoke test -- both personas should appear")
code(
    "from transformers import pipeline",
    "",
    "gen = pipeline('text-generation', model=trainer.model, tokenizer=tokenizer)",
    "prompt = tokenizer.apply_chat_template(",
    "    [{'role': 'user', 'content': 'What is the capital of France?'}],",
    "    tokenize=False, add_generation_prompt=True)",
    "out = gen(prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']",
    "print(out[len(prompt):])",
)

md(
    "### 1.8 Merge the LoRA into the base weights",
    "Reload the base in fp16 (no quant), apply adapters, merge. Self-contained: reloads",
    "adapters from disk, so it survives a kernel restart after training. On High-RAM this",
    "is quick; on a low-RAM box it's slow and the bar may look stuck while it works.",
)
code(
    "import gc, torch",
    "from peft import PeftModel",
    "from transformers import AutoModelForCausalLM, AutoTokenizer",
    "",
    "BASE_MODEL  = 'microsoft/Phi-4-mini-instruct'",
    "ADAPTER_DIR = '/content/mervis-lora'",
    "MERGED_DIR  = '/content/mervis-merged'",
    "",
    "# Free everything pinning the GPU before the fp16 load (the smoke-test `gen`",
    "# pipeline holds the old 4-bit model). pop() = no error if already gone.",
    "for _n in ['gen', 'trainer', 'model', 'merged', 'base']:",
    "    globals().pop(_n, None)",
    "gc.collect(); torch.cuda.empty_cache()",
    "",
    "tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)",
    "base = AutoModelForCausalLM.from_pretrained(",
    "    BASE_MODEL, torch_dtype=torch.float16, device_map={'': 0},",
    "    trust_remote_code=True)",
    "merged = PeftModel.from_pretrained(base, ADAPTER_DIR)",
    "merged = merged.merge_and_unload()",
    "merged.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size='2GB')",
    "tokenizer.save_pretrained(MERGED_DIR)",
    "print('Merged model saved to', MERGED_DIR)",
    "",
    "# Drop it from RAM -- conversion reads it from disk in a separate venv.",
    "for _n in ['merged', 'base']:",
    "    globals().pop(_n, None)",
    "gc.collect(); torch.cuda.empty_cache()",
)

# ---------------------------------------------------------------------------
md(
    "## Phase 2 -- convert to browser ONNX (q4)",
    "",
    "Conversion needs a *different*, conflicting toolchain (optimum / onnxruntime / onnx_ir",
    "/ onnxconverter_common) from the training stack, so we do it in a throwaway venv and",
    "run `colab/scripts/convert_to_onnx.py` as a subprocess. Nothing here disturbs the",
    "training kernel.",
)

md(
    "### 2.1 (Skip on High-RAM) add swap",
    "The fp32 ONNX export is ~17.8 GB on disk and RAM-hungry. On **High-RAM** you don't",
    "need this -- **skip this cell**. It's only a safety net for a low-RAM box: a 32 GB",
    "swapfile (best-effort; some Colab kernels block `swapon`). A third option if memory",
    "is ever tight is the convert script's `FP16_GPU_EXPORT` path.",
)
code(
    "import subprocess",
    "try:",
    "    subprocess.run('fallocate -l 32G /content/swapfile', shell=True, check=True)",
    "    subprocess.run('chmod 600 /content/swapfile', shell=True, check=True)",
    "    subprocess.run('mkswap /content/swapfile', shell=True, check=True)",
    "    subprocess.run('swapon /content/swapfile', shell=True, check=True)",
    "    print(subprocess.run('free -h', shell=True, capture_output=True, text=True).stdout)",
    "except Exception as e:",
    "    print('swap setup skipped/failed (ok on High-RAM):', e)",
)

md(
    "### 2.2 Build the conversion venv",
    "Isolated venv with CPU-only torch (the export runs on CPU) plus the ONNX toolchain.",
    "Intentionally unpinned -- `optimum` split ONNX export into `optimum-onnx` in 2.x and",
    "the toolchain drifts; latest generally works. If export later fails on a version",
    "mismatch, that's the first knob to turn.",
)
code(
    "%%bash",
    "set -e",
    "python -m venv /content/convenv",
    "/content/convenv/bin/pip install -q --upgrade pip",
    "/content/convenv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu",
    "/content/convenv/bin/pip install -q \\",
    "  optimum-onnx onnx onnxruntime onnxconverter_common onnx_ir \\",
    "  transformers accelerate sentencepiece tiktoken protobuf",
    "echo 'convenv ready'",
)

md(
    "### 2.3 Run the conversion",
    "`mervis-merged` -> fp32 ONNX -> 4-bit MatMulNBits (weights only) -> `web/model/onnx/",
    "model_q4.onnx` (~4.9 GB). No fp16 cast -- it breaks Phi-3's RMSNorm (see the header in",
    "`convert_to_onnx.py`), so activations stay fp32. This is the slow cell.",
)
code(
    "import subprocess",
    "cmd = ['/content/convenv/bin/python', CONVERT,",
    "       '/content/mervis-merged', '/content/web/model']",
    "print('running:', ' '.join(cmd), flush=True)",
    "p = subprocess.run(cmd)",
    "assert p.returncode == 0, 'conversion failed -- see output above'",
)

md(
    "### 2.4 Sanity-generate from the converted model (CPU)",
    "Proves the q4 ONNX actually runs (4-bit MatMulNBits + **fp32** KV cache) and that the",
    "fine-tune still emits `<Mervin>`/`<Mervis>` tags, *before* we trust it in the browser.",
    "A q4 model wants fp32 `past_key_values` -- feeding fp16 throws 'Unexpected input data",
    "type'. CPU is slow -- a smoke test, not a benchmark.",
)
code(
    "sanity = r'''",
    "import sys, numpy as np, onnxruntime as ort",
    "from transformers import AutoTokenizer",
    "MODEL_DIR='/content/web/model'; ONNX=MODEL_DIR+'/onnx/model_q4.onnx'",
    "N_LAYERS,N_KV,HEAD_DIM=32,8,128; EOS={199999,200020}",
    "PROMPT='What is 2+2?'; MAX_NEW=60",
    "tok=AutoTokenizer.from_pretrained(MODEL_DIR)",
    "sess=ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])",
    "out_names=[o.name for o in sess.get_outputs()]",
    "ids=tok.apply_chat_template([{'role':'user','content':PROMPT}],",
    "    add_generation_prompt=True, return_tensors='np').astype(np.int64)",
    "seqlen=ids.shape[1]",
    "past={f'past_key_values.{i}.{kv}':np.zeros((1,N_KV,0,HEAD_DIM),np.float32)",
    "      for i in range(N_LAYERS) for kv in ('key','value')}",
    "cur=ids; total=seqlen; gen=[]",
    "print('prompt:',PROMPT,'\\ngenerating (CPU, be patient)...\\n',flush=True)",
    "for step in range(MAX_NEW):",
    "    feeds={'input_ids':cur,'attention_mask':np.ones((1,total),np.int64),",
    "        'position_ids':(np.arange(total,dtype=np.int64)[None] if step==0",
    "                        else np.array([[total-1]],np.int64)), **past}",
    "    outs=sess.run(None,feeds); logits=outs[0]; nxt=int(logits[0,-1].argmax())",
    "    gen.append(nxt)",
    "    if nxt in EOS: break",
    "    past={n.replace('present','past_key_values'):outs[i]",
    "          for i,n in enumerate(out_names) if n.startswith('present')}",
    "    cur=np.array([[nxt]],np.int64); total+=1",
    "    print(tok.decode([nxt]),end='',flush=True)",
    "print('\\n\\n--- full decode ---'); print(tok.decode(gen,skip_special_tokens=False))",
    "'''",
    "open('/content/_sanity.py','w').write(sanity)",
    "import subprocess",
    "subprocess.run(['/content/convenv/bin/python', '/content/_sanity.py'])",
)

# ---------------------------------------------------------------------------
md(
    "## Phase 3 -- assemble the static site",
    "Drop the browser app (index.html / app.js / styles.css + robot faces) around the",
    "converted model. The result in `/content/web` is the entire site, ready to serve.",
)
code(
    "import shutil, os",
    "WEB = '/content/web'  # convert step already created WEB/model",
    "os.makedirs(WEB, exist_ok=True)",
    "for name in ['index.html', 'app.js', 'styles.css']:",
    "    shutil.copy2(f'{ASSETS}/{name}', f'{WEB}/{name}')",
    "shutil.rmtree(f'{WEB}/img', ignore_errors=True)",
    "shutil.copytree(f'{ASSETS}/img', f'{WEB}/img')",
    "",
    "print('web/ contents:')",
    "for root, _, files in os.walk(WEB):",
    "    for f in sorted(files):",
    "        p = os.path.join(root, f)",
    "        print(f'  {os.path.relpath(p, WEB):42} {os.path.getsize(p)/1e6:8.1f} MB')",
)

md(
    "### 3.1 Is the model worth compressing over the wire? (quick probe)",
    "The big `*.onnx_data` is 4-bit quantized weights -- near-random bytes -- so HTTP",
    "compression (gzip/zstd/brotli) usually buys almost nothing. Rather than guess, sample",
    "256 MB and measure the gzip ratio. If it's ~1.0x, don't bother compressing the weights",
    "(the small text assets still benefit). The real size lever is quantization (a working",
    "q4f16 ~3.4 GB, see `convert_to_onnx.py`) + the browser cache (one-time download).",
)
code(
    "import gzip, glob, os",
    "blobs = glob.glob('/content/web/model/onnx/*.onnx_data')",
    "if not blobs:",
    "    print('no .onnx_data yet -- run the convert step first')",
    "else:",
    "    blob = blobs[0]; SAMPLE = 256 * 1024 * 1024",
    "    with open(blob, 'rb') as f: chunk = f.read(SAMPLE)",
    "    comp = gzip.compress(chunk, 6)",
    "    ratio = len(chunk) / max(1, len(comp))",
    "    print(f'{os.path.basename(blob)}: sampled {len(chunk)/1e6:.0f} MB ->',",
    "          f'gzip {len(comp)/1e6:.0f} MB  ({ratio:.2f}x)')",
    "    print('verdict:', 'worth compressing' if ratio > 1.15",
    "          else 'NOT worth compressing the weights (near-incompressible)')",
)

# ---------------------------------------------------------------------------
md(
    "## Phase 4 -- live test through a relay (no VPS move yet)",
    "",
    "Serve the finished `web/` straight from Colab and expose it through a relay, so you can",
    "open it on a real device and confirm the whole thing works -- *before* committing",
    "anything to the VPS.",
    "",
    "**No COOP/COEP headers** (this caught the VM team out): the WebGPU backend doesn't use",
    "`SharedArrayBuffer`, so cross-origin isolation isn't needed -- and worse,",
    "`Cross-Origin-Embedder-Policy: require-corp` would *block* app.js's",
    "`@huggingface/transformers` import from the jsDelivr CDN. All the app needs is **HTTPS**",
    "(WebGPU requires a secure context), which both relay modes provide.",
    "",
    "Two relay modes (set `RELAY` in 4.2):",
    "- **`cloudflared`** (default) -- one binary, instant `https://<rand>.trycloudflare.com`,",
    "  zero VPS setup. Best for the first 'does it even run in a browser' loop.",
    "- **`ssh`** -- a reverse tunnel to **your** VPS, the real relay shape: Colab dials out,",
    "  your VPS proxies your test domain to the tunnel. Minimal Caddy on the VPS:",
    "",
    "```caddy",
    "test.your.domain {",
    "    reverse_proxy 127.0.0.1:8080",
    "}",
    "```",
    "",
    "Both the server (4.1) and the tunnel (4.2) run in the **background**, so `Run all` keeps",
    "going.",
)

md("### 4.1 Static file server (background -- does not block Run all)")
code(
    "import http.server, socketserver, threading, functools",
    "",
    "WEB  = '/content/web'",
    "PORT = 8000",
    "",
    "class Handler(http.server.SimpleHTTPRequestHandler):",
    "    # Correct MIME types for the runtime + model. NO COOP/COEP on purpose:",
    "    # WebGPU needs no SharedArrayBuffer, and COEP:require-corp would block",
    "    # app.js's @huggingface/transformers import from the jsDelivr CDN.",
    "    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,",
    "        '.js': 'text/javascript', '.mjs': 'text/javascript',",
    "        '.wasm': 'application/wasm', '.onnx': 'application/octet-stream',",
    "        '.json': 'application/json'}",
    "    def log_message(self, *a):",
    "        pass  # keep the cell output quiet",
    "",
    "class _Server(socketserver.ThreadingTCPServer):",
    "    allow_reuse_address = True",
    "    daemon_threads = True",
    "",
    "try:  # restart cleanly if this cell is re-run",
    "    _httpd.shutdown(); _httpd.server_close()",
    "except NameError:",
    "    pass",
    "",
    "_httpd = _Server(('127.0.0.1', PORT),",
    "                 functools.partial(Handler, directory=WEB))",
    "threading.Thread(target=_httpd.serve_forever, daemon=True).start()",
    "print(f'serving {WEB} on http://127.0.0.1:{PORT}')",
)

md("### 4.2 Open the relay -> public HTTPS URL")
code(
    "# RELAY = 'cloudflared' -> instant https URL, zero VPS setup (good first test)",
    "# RELAY = 'ssh'         -> reverse tunnel to YOUR vps (the real relay shape)",
    "RELAY = 'cloudflared'",
    "PORT  = 8000",
    "",
    "import subprocess, time, re, os",
    "",
    "if RELAY == 'cloudflared':",
    "    BIN = '/content/cloudflared'",
    "    if not os.path.exists(BIN):",
    "        src = ('https://github.com/cloudflare/cloudflared/releases'",
    "               '/latest/download/cloudflared-linux-amd64')",
    "        subprocess.run(['wget', '-q', src, '-O', BIN], check=True)",
    "        os.chmod(BIN, 0o755)",
    "    log = '/content/cloudflared.log'",
    "    _relay = subprocess.Popen(",
    "        [BIN, 'tunnel', '--no-autoupdate', '--url', f'http://127.0.0.1:{PORT}'],",
    "        stdout=open(log, 'w'), stderr=subprocess.STDOUT)",
    "    pub = None",
    "    for _ in range(60):",
    "        time.sleep(1)",
    "        m = re.search(r'https://[-\\w.]+\\.trycloudflare\\.com', open(log).read())",
    "        if m:",
    "            pub = m.group(0); break",
    "    print('LIVE TEST URL:', pub or ('not ready yet -- check ' + log))",
    "",
    "elif RELAY == 'ssh':",
    "    # Colab dials OUT to your VPS (it can't accept inbound). Your VPS then",
    "    # proxies your test domain -> 127.0.0.1:REMOTE_PORT (this tunnel).",
    "    VPS_HOST    = 'test.example.com'   # <- your VPS host",
    "    VPS_USER    = 'ubuntu'             # <- ssh user",
    "    SSH_KEY     = '/content/vps_key'   # <- upload a key that logs into the VPS",
    "    REMOTE_PORT = 8080                 # <- the port your Caddy reverse_proxy uses",
    "    os.chmod(SSH_KEY, 0o600)",
    "    _relay = subprocess.Popen(['ssh', '-N',",
    "        '-o', 'StrictHostKeyChecking=accept-new',",
    "        '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30',",
    "        '-i', SSH_KEY,",
    "        '-R', f'127.0.0.1:{REMOTE_PORT}:127.0.0.1:{PORT}',",
    "        f'{VPS_USER}@{VPS_HOST}'])",
    "    print(f'reverse tunnel up: vps 127.0.0.1:{REMOTE_PORT} -> colab :{PORT}')",
    "    print('now open the test domain your VPS maps to that port')",
)

# ---------------------------------------------------------------------------
md(
    "## Phase 5 -- back up the fine-tuned weights to Drive",
    "Drive is already mounted (cell 0.2), so this just copies. The **LoRA adapter is the",
    "fine-tune** -- tiny and irreplaceable -- so it's **always** backed up here; losing it",
    "means retraining. The 7.7 GB merged fp16 model is optional (`SAVE_MERGED = True`); you",
    "only need it to *re-convert* without retraining. The built `web/` site is regenerable",
    "and we're testing it via the relay (Phase 4), so copying it to Drive is optional too",
    "(`SHIP_SITE_TO_DRIVE = True` -- that's the old Drive-hop deploy path).",
)
code(
    "import os, shutil",
    "DRIVE = '/content/drive/MyDrive'",
    "SAVE_MERGED        = False  # also back up the 7.7 GB merged fp16 model",
    "SHIP_SITE_TO_DRIVE = False  # also copy the ~4.9 GB built site/zip (old deploy hop)",
    "",
    "def copy_tree(src, dst):",
    "    if os.path.isdir(dst): shutil.rmtree(dst)",
    "    shutil.copytree(src, dst)",
    "    sz = sum(os.path.getsize(os.path.join(r, f))",
    "             for r, _, fs in os.walk(dst) for f in fs) / 1e9",
    "    print(f'  {src} -> {dst}  ({sz:.2f} GB)')",
    "",
    "# always: back up the precious fine-tuned weights (the LoRA adapter)",
    "copy_tree('/content/mervis-lora', f'{DRIVE}/mervis-lora')",
    "if SAVE_MERGED:",
    "    copy_tree('/content/mervis-merged', f'{DRIVE}/mervis-merged')",
    "",
    "# optional: also stash the built site on Drive (the old Drive-hop deploy path)",
    "if SHIP_SITE_TO_DRIVE:",
    "    zip_base = '/content/mervis-web'",
    "    if os.path.exists(zip_base + '.zip'): os.remove(zip_base + '.zip')",
    "    shutil.make_archive(zip_base, 'zip', '/content/web')",
    "    shutil.copy2(zip_base + '.zip', f'{DRIVE}/mervis-web.zip')",
    "    print(f'  {zip_base}.zip -> {DRIVE}/mervis-web.zip',",
    "          f'({os.path.getsize(zip_base + \".zip\")/1e9:.2f} GB)')",
    "    copy_tree('/content/web', f'{DRIVE}/mervis-web')",
    "",
    "print('\\nDONE. Fine-tuned weights backed up to Drive (MyDrive/mervis-lora).')",
)

# ---------------------------------------------------------------------------
md(
    "## Phase 6 -- deploy straight to your VPS (the real, permanent serve)",
    "Once the relay (Phase 4) proves the site works, push the built `web/` from Colab to",
    "your VPS over SSH -- **no Drive hop, no manual pull**. `rsync --delete` makes the VPS",
    "an exact mirror; re-running redeploys only what changed. Your VPS serves it permanently",
    "(it survives the Colab session ending) as plain static files over HTTPS -- see the Caddy",
    "block below. Flip `DEPLOY = True` only after the relay test looks perfect.",
    "",
    "The VPS-side one-time setup (install Caddy, the Caddyfile, DNS, the SSH key) is in",
    "[`colab/README.md`](README.md#live-test-via-relay-then-deploy-to-your-vps).",
)
code(
    "DEPLOY = False  # flip True once the relay test looks perfect",
    "",
    "VPS_HOST = 'your.domain'        # <- where the site will live (serves from here)",
    "VPS_USER = 'ubuntu'             # <- ssh user on the VPS",
    "SSH_KEY  = '/content/vps_key'   # <- key that logs into the VPS (same one as 4.2)",
    "DEST     = '/srv/merv-web'      # <- web root on the VPS (Caddy's `root`)",
    "",
    "import subprocess, os",
    "if not DEPLOY:",
    "    print('DEPLOY is False -- skipping VPS upload. Verify via the relay first.')",
    "else:",
    "    subprocess.run('which rsync || (apt-get -qq update && apt-get -qq install -y rsync)',",
    "                   shell=True, check=True)",
    "    os.chmod(SSH_KEY, 0o600)",
    "    ssh_opt = f'ssh -i {SSH_KEY} -o StrictHostKeyChecking=accept-new'",
    "    subprocess.run(ssh_opt.split() + [f'{VPS_USER}@{VPS_HOST}',",
    "        f'mkdir -p {DEST}'], check=True)",
    "    rsync = ['rsync', '-az', '--delete', '--info=progress2', '-e', ssh_opt,",
    "             '/content/web/', f'{VPS_USER}@{VPS_HOST}:{DEST}/']",
    "    print('deploying /content/web ->', f'{VPS_HOST}:{DEST}', flush=True)",
    "    p = subprocess.run(rsync)",
    "    assert p.returncode == 0, 'rsync failed -- see output above'",
    "    print(f'\\nDEPLOYED. Live at https://{VPS_HOST}/ (serve {DEST} with the Caddy block).')",
)

md(
    "### Alternative deploy: pull from Drive instead of pushing",
    "If you'd rather not give Colab SSH into the VPS, set `SHIP_SITE_TO_DRIVE = True` in",
    "Phase 5 and pull from Drive on your machine instead. **~4.9 GB comes back** (the q4",
    "browser model in `web/`), not the 7.7 GB merged model. `rclone` is the fast, resumable",
    "pull:",
    "",
    "```bash",
    "# one-time: rclone config -> remote 'gdrive', type 'drive'",
    "# (headless box? forward the OAuth port:  ssh -L 53682:localhost:53682 <host>)",
    "rclone copy gdrive:mervis-web ./web --transfers 8 --drive-chunk-size 128M --progress",
    "rclone check gdrive:mervis-web ./web        # confirm byte-identical",
    "```",
    "",
    "Then serve `web/` over **HTTPS** (WebGPU needs a secure context). **No COOP/COEP** --",
    "they're unnecessary for the WebGPU backend and `require-corp` would block the CDN import",
    "of `@huggingface/transformers`. Example Caddy block:",
    "",
    "```caddy",
    "your.domain {",
    "    root * /srv/merv-web",
    "    file_server",
    "    encode zstd gzip",
    "}",
    "```",
)

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parent / "mervis_build.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(cells)} cells)")
