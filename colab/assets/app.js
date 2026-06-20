// Mervin & Mervis — in-browser chat on a fine-tuned Phi-4-mini (ONNX q4f16, WebGPU).
// Everything runs client-side; the model is served same-origin from ./model/.

import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";

// Load weights from our own origin, never the HF Hub.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "./"; // model id "model" -> ./model/

const MODEL_ID = "model";

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
  if (!navigator.gpu) {
    els.noWebgpu.hidden = false;
    return;
  }
  els.loadBtn.hidden = true;
  els.loadStatus.hidden = false;
  try {
    tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
    model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      dtype: "q4f16",
      device: "webgpu",
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

  const input_ids = tokenizer.apply_chat_template(history, {
    add_generation_prompt: true,
    return_tensor: true,
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
    input_ids,
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
if (!navigator.gpu) els.noWebgpu.hidden = false;

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
