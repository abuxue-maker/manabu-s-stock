/* Speech-to-text engine (Whisper, runs inside the phone; nothing is sent anywhere).
   Loaded as a module worker so the page never freezes while the model works. */
const LIB = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

let lib = null;
let asr = null;
let device = 'wasm';
let modelId = '';
let loadToken = 0;

const post = (m) => self.postMessage(m);

async function getLib() {
  if (!lib) {
    lib = await import(LIB);
    lib.env.allowLocalModels = false;
    lib.env.useBrowserCache = true;      // model files are kept in the phone's cache for offline use
  }
  return lib;
}

function progressReporter(token) {
  const files = new Map();
  let lastPost = 0;
  return (p) => {
    if (token !== loadToken || !p) return;
    if (p.file && typeof p.total === 'number' && p.total > 0 && (p.status === 'progress' || p.status === 'done')) {
      files.set(p.file, { loaded: p.status === 'done' ? p.total : p.loaded || 0, total: p.total });
    }
    const now = Date.now();
    if (now - lastPost < 150 && p.status === 'progress') return;
    lastPost = now;
    let loaded = 0;
    let total = 0;
    files.forEach((f) => { loaded += f.loaded; total += f.total; });
    post({ type: 'progress', loaded, total, pct: total ? Math.min(99, Math.floor((loaded / total) * 100)) : 0 });
  };
}

async function canUseGpu() {
  try {
    if (!self.navigator || !self.navigator.gpu) return false;
    const ad = await self.navigator.gpu.requestAdapter();
    return !!ad;
  } catch (e) { return false; }
}

async function build(model, dev, token) {
  const { pipeline } = await getLib();
  const opts = { device: dev, progress_callback: progressReporter(token) };
  if (dev === 'webgpu') opts.dtype = { encoder_model: 'fp32', decoder_model_merged: 'q4' };
  const p = await pipeline('automatic-speech-recognition', model, opts);
  return p;
}

function run(p, audio) {
  return p(audio, { language: 'japanese', task: 'transcribe', return_timestamps: false });
}

async function dispose() {
  const old = asr;
  asr = null;
  try { if (old && old.dispose) await old.dispose(); } catch (e) { /* ignore */ }
}

async function load(model, preferGpu) {
  const token = ++loadToken;
  await dispose();
  post({ type: 'loading', model });
  let dev = preferGpu && (await canUseGpu()) ? 'webgpu' : 'wasm';
  for (;;) {
    try {
      const p = await build(model, dev, token);
      if (token !== loadToken) { try { await p.dispose(); } catch (e) { /* ignore */ } return; }
      const t0 = Date.now();
      await run(p, new Float32Array(16000));     // first run compiles things: do it now, not at the first sentence
      if (token !== loadToken) return;
      asr = p;
      device = dev;
      modelId = model;
      post({ type: 'ready', device: dev, model, warmMs: Date.now() - t0 });
      return;
    } catch (err) {
      if (token !== loadToken) return;
      if (dev === 'webgpu') {                    // graphics route failed: use the plain route
        post({ type: 'note', message: 'webgpu: ' + (err && err.message ? err.message : err) });
        dev = 'wasm';
        continue;
      }
      post({ type: 'error', stage: 'load', message: String(err && err.message ? err.message : err) });
      return;
    }
  }
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'load') {
    load(m.model, m.gpu !== false);
  } else if (m.type === 'run') {
    if (!asr) { post({ type: 'error', stage: 'run', id: m.id, message: 'engine not ready' }); return; }
    const t0 = Date.now();
    try {
      let out;
      try {
        out = await run(asr, m.audio);
      } catch (err) {
        if (device !== 'webgpu') throw err;
        post({ type: 'note', message: 'webgpu run failed, switching: ' + (err && err.message ? err.message : err) });
        await load(modelId, false);
        if (!asr) throw err;
        out = await run(asr, m.audio);
      }
      const text = Array.isArray(out) ? out.map((o) => o.text).join(' ') : out.text;
      post({ type: 'result', id: m.id, text: text || '', ms: Date.now() - t0 });
    } catch (err) {
      post({ type: 'error', stage: 'run', id: m.id, message: String(err && err.message ? err.message : err) });
    }
  }
};

post({ type: 'boot' });
