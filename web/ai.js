'use strict';

// Loads the typed-search WASM engine and the strongest (v5) table set, then
// exposes a small synchronous move-selection API to the game UI.
//
// Rank indexing matches the C++ engine: index 0='3' … 10='K', 11='A', 12='2'.
const Big2 = (() => {
  const FILES = ['eval_extended', 'eval_main', 'eval_fallback',
                 'mp_disc', 'mp_main', 'mp_fallback'];
  // Sum of the v5 table file sizes (bytes) — used for the load progress bar.
  const TOTAL_BYTES = 2734676 + 919580 + 55520 + 43215020 + 8656352 + 195772;

  let Module = null;
  let ready = false;

  async function fetchToMemfs(url, baseLoaded, onProgress) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('failed to fetch ' + url + ': ' + resp.status);
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (onProgress) onProgress(baseLoaded + received);
    }
    const buf = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    return buf;
  }

  // onProgress(fraction in [0,1]); resolves to true on success.
  async function init(onProgress) {
    Module = await Big2AI();
    try { Module.FS.mkdir('/tables'); } catch (e) { /* exists */ }

    let loaded = 0;
    for (const f of FILES) {
      const buf = await fetchToMemfs('tables/' + f + '.bin', loaded,
        (b) => onProgress && onProgress(Math.min(b / TOTAL_BYTES, 0.999)));
      loaded += buf.length;
      Module.FS.writeFile('/tables/' + f + '.bin', buf);
    }

    const n = Module.ccall('ts_load_tables', 'number', [], []);
    // Reclaim the ~54MB of staged file bytes; tables now live in C++ maps.
    for (const f of FILES) { try { Module.FS.unlink('/tables/' + f + '.bin'); } catch (e) {} }
    ready = n > 0;
    if (onProgress) onProgress(1);
    return ready;
  }

  // hand, discard, trick: 13-element rank-count arrays. trick is the cards on
  // the table to beat (all zeros = we lead). Returns { moveId, cards } where
  // cards is the chosen move's 13-element rank-count vector (all zeros = pass).
  function selectMove(hand, discard, oppCount, trick) {
    const toHeap = (arr) => {
      const p = Module._malloc(13 * 4);
      Module.HEAP32.set(Int32Array.from(arr), p >> 2);
      return p;
    };
    const hp = toHeap(hand), dp = toHeap(discard), tp = toHeap(trick);
    const op = Module._malloc(13 * 4);
    const moveId = Module.ccall('ts_select_move', 'number',
      ['number', 'number', 'number', 'number', 'number'],
      [hp, dp, oppCount, tp, op]);
    const cards = Array.from(Module.HEAP32.subarray(op >> 2, (op >> 2) + 13));
    [hp, dp, tp, op].forEach((p) => Module._free(p));
    return { moveId, cards };
  }

  return { init, selectMove, isReady: () => ready };
})();
