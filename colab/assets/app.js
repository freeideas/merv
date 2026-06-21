// Mervin & Mervis — in-browser chat on a fine-tuned Phi-4-mini (ONNX q4, WebGPU).
// Everything runs client-side; the model is served same-origin from ./model/.

// Library version is overridable for A/B testing WebGPU output quality across
// releases (newer onnxruntime-web has WebGPU q4 accuracy fixes): ?tjs=3.8.1
const _tjs = new URLSearchParams(location.search).get("tjs");
const TJS_VERSION = _tjs && /^[\w.\-]+$/.test(_tjs) ? _tjs : "3.3.3";
const { AutoTokenizer, AutoModelForCausalLM, TextStreamer, env } = await import(
  `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TJS_VERSION}`
);

// Load weights from our own origin, never the HF Hub.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "./"; // model id "model" -> ./model/

const MODEL_ID = "model";
// Bump whenever the model files change (re-quantize, re-shard, new manifest) so
// returning visitors evict their cached copies instead of loading a stale mix
// (e.g. an old cached graph that references a shard the new manifest dropped).
const MODEL_VERSION = "2026-06-21-embsplit-2shard";

// Runtime override for A/B testing: ?device=wasm runs the SAME q4 model on the
// CPU/WASM backend (slow, but numerically matches local inference) so you can
// compare answer quality against the default WebGPU backend on the same machine.
const DEVICE = new URLSearchParams(location.search).get("device") || "webgpu";

const els = {
  loadBtn: document.getElementById("load-btn"),
  loadStatus: document.getElementById("load-status"),
  barFill: document.getElementById("bar-fill"),
  loadText: document.getElementById("load-text"),
  noWebgpu: document.getElementById("no-webgpu"),
  loader: document.getElementById("loader"),
  chat: document.getElementById("chat"),
  composer: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
};

let tokenizer = null;
let model = null;
const history = []; // [{ role: "user"|"assistant", content }]

// ---- model loading -------------------------------------------------------

function onProgress(p) {
  if (p.status === "progress" && p.total) {
    const pct = Math.round((p.loaded / p.total) * 100);
    els.barFill.style.width = pct + "%";
    els.loadText.textContent = `${p.file} — ${pct}%`;
  } else if (p.status === "done") {
    els.loadText.textContent = `${p.file} ready`;
  } else if (p.status === "ready") {
    els.loadText.textContent = "Warming up WebGPU…";
  }
}

async function loadModel() {
  if (DEVICE === "webgpu" && !navigator.gpu) {
    els.noWebgpu.hidden = false;
    return;
  }
  els.loadBtn.hidden = true;
  els.loadStatus.hidden = false;
  try {
    // Transformers.js caches model files in the "transformers-cache" Cache
    // Storage and serves them back WITHOUT revalidation. When the model content
    // changes at the same URLs (e.g. re-sharding), a returning visitor would load
    // a stale/mismatched mix. Evict the old cache whenever MODEL_VERSION changes.
    try {
      if (localStorage.getItem("merv_model_version") !== MODEL_VERSION) {
        if (self.caches) await caches.delete("transformers-cache");
        localStorage.setItem("merv_model_version", MODEL_VERSION);
      }
    } catch (_) {
      /* private mode / storage disabled — proceed without the cache-bust */
    }

    tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
    // The weights are sharded into <2 GB files: V8 caps a single ArrayBuffer at
    // ~2 GB, and the default loader fetches the whole *.onnx_data into ONE buffer
    // (so a 3.6 GB sidecar can never load). We fetch the manifest and hand ORT
    // each shard separately via session_options.externalData -- each its own
    // buffer, all mounted into the WASM heap (which holds the full ~3.6 GB fine).
    // no-store: the manifest is tiny and must reflect the CURRENT shard set; a
    // stale HTTP-cached copy here desyncs externalData from the deployed shards.
    const manifest = await (
      await fetch("./model/onnx/external_data_manifest.json", { cache: "no-store" })
    ).json();
    const externalData = manifest.shards.map((name) => ({
      path: name, // must match the graph's external-data location string
      data: `onnx/${name}`, // fetched relative to the model dir (./model/)
    }));
    model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      dtype: "q4", // q4f16 is broken for Phi-3 RMSNorm; see convert_to_onnx.py
      device: DEVICE, // "webgpu" (default) or ?device=wasm for CPU A/B testing
      session_options: { externalData },
      progress_callback: onProgress,
    });
  } catch (err) {
    els.loadText.textContent = "Load failed: " + err.message;
    throw err;
  }
  els.loader.hidden = true;
  els.chat.hidden = false;
  els.composer.hidden = false;
  els.input.focus();
}

// ---- tag splitting (Phase 3) --------------------------------------------

// Pull the inside of <Tag>…</Tag>, tolerating a half-open tag while streaming.
function extractTag(text, tag) {
  const open = `<${tag}>`;
  const close = `</${tag}>`;
  const start = text.indexOf(open);
  if (start === -1) return null;
  const from = start + open.length;
  const end = text.indexOf(close, from);
  return (end === -1 ? text.slice(from) : text.slice(from, end)).trim();
}

function splitPersonas(text) {
  const mervin = extractTag(text, "Mervin");
  const mervis = extractTag(text, "Mervis");
  if (mervin === null && mervis === null) return null; // no tags yet/ever
  return { mervin: mervin ?? "", mervis: mervis ?? "" };
}

// ---- rendering -----------------------------------------------------------

function addUserBubble(text) {
  const row = document.createElement("div");
  row.className = "msg user";
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = text;
  row.appendChild(b);
  els.chat.appendChild(row);
  scrollToEnd();
}

// Returns an updater(fullText, done) that renders the two persona bubbles live.
function addAssistantBubbles() {
  const wrap = document.createElement("div");
  wrap.className = "assistant";

  const make = (who, name, img) => {
    const row = document.createElement("div");
    row.className = `persona ${who}`;
    row.innerHTML =
      `<img src="./img/${img}" alt="${name}" />` +
      `<div><div class="name">${name}</div><div class="bubble cursor"></div></div>`;
    wrap.appendChild(row);
    return { row, bubble: row.querySelector(".bubble") };
  };

  const mervin = make("mervin", "Mervin 🤖💧", "bot-sad.png");
  const mervis = make("mervis", "Mervis 🤖✨", "bot-happy.png");
  const fallback = document.createElement("div");
  fallback.className = "persona";
  fallback.innerHTML = `<div><div class="bubble cursor"></div></div>`;

  els.chat.appendChild(wrap);
  scrollToEnd();

  return function update(fullText, done) {
    const split = splitPersonas(fullText);
    if (split) {
      if (fallback.parentNode) fallback.remove();
      if (!mervin.row.parentNode) wrap.appendChild(mervin.row);
      if (!mervis.row.parentNode) wrap.appendChild(mervis.row);
      mervin.bubble.textContent = split.mervin;
      mervis.bubble.textContent = split.mervis;
    } else {
      // No tags arrived — degrade gracefully to one plain bubble.
      mervin.row.remove();
      mervis.row.remove();
      if (!fallback.parentNode) wrap.appendChild(fallback);
      fallback.querySelector(".bubble").textContent = fullText;
    }
    if (done) {
      wrap.querySelectorAll(".cursor").forEach((n) => n.classList.remove("cursor"));
    }
    scrollToEnd();
  };
}

function scrollToEnd() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

// ---- generation ----------------------------------------------------------

async function generate(userText) {
  history.push({ role: "user", content: userText });
  addUserBubble(userText);
  const update = addAssistantBubbles();

  // return_dict gives us BOTH input_ids and attention_mask. The ONNX graph
  // requires attention_mask (and derives position_ids from it), so passing
  // input_ids alone fails at generate() with:
  //   "Missing the following inputs: attention_mask, position_ids".
  const inputs = tokenizer.apply_chat_template(history, {
    add_generation_prompt: true,
    return_dict: true,
  });

  let full = "";
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk) => {
      full += chunk;
      update(full, false);
    },
  });

  await model.generate({
    ...inputs,
    max_new_tokens: 512,
    do_sample: true,
    temperature: 0.7,
    top_p: 0.9,
    streamer,
  });

  update(full, true);
  history.push({ role: "assistant", content: full });
}

// ---- wiring --------------------------------------------------------------

els.loadBtn.addEventListener("click", loadModel);
if (DEVICE === "webgpu" && !navigator.gpu) els.noWebgpu.hidden = false;

els.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = els.input.value.trim();
  if (!text || !model) return;
  els.input.value = "";
  els.input.style.height = "auto";
  els.send.disabled = true;
  els.input.disabled = true;
  try {
    await generate(text);
  } finally {
    els.send.disabled = false;
    els.input.disabled = false;
    els.input.focus();
  }
});

// grow textarea + submit on Enter (Shift+Enter = newline)
els.input.addEventListener("input", () => {
  els.input.style.height = "auto";
  els.input.style.height = els.input.scrollHeight + "px";
});
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});
