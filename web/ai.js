'use strict';
// AZ neural-net CPU brain for the web game. Replaces the old typed_search WASM
// path: spawns worker.js (ONNX Runtime Web + plain-tree MCTS) and exposes a
// small async move API to cardgame.js via window.Big2.
//
// Rank index convention (matches the engine): 0='3' … 10='K', 11='A', 12='2'.
import { MoveTable, kPASS } from './engine.js';

const Big2 = (() => {
  let tbl = null;          // MoveTable (main-thread: legality + cards→id map)
  let cardsToId = null;    // "c0,c1,…,c12" -> move id (unique for non-pass)
  let worker = null;
  let ready = false;
  let seq = 0;
  const pending = new Map();  // request id -> {resolve, reject}

  async function init(onProgress) {
    if (onProgress) onProgress(0.1);
    const moves = await (await fetch('./az_moves.json')).json();
    tbl = new MoveTable(moves);
    cardsToId = new Map();
    for (const m of tbl.moves) if (m.id !== kPASS) cardsToId.set(m.cards.join(','), m.id);
    if (onProgress) onProgress(0.4);

    worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (ev) => {
      const msg = ev.data;
      if (msg.type === 'ready') { ready = true; if (onProgress) onProgress(1); return; }
      if (msg.type === 'move' && msg.reqId != null) {
        const p = pending.get(msg.reqId);
        if (p) { pending.delete(msg.reqId); p.resolve(msg.moveId); }
        return;
      }
      if (msg.type === 'error') {
        console.error('worker error:', msg.error);
        for (const [, p] of pending) p.reject(new Error(msg.error));
        pending.clear();
      }
    };
    // Wait for the worker (model + tables loaded) to report ready.
    await new Promise((res) => {
      const t = setInterval(() => { if (ready) { clearInterval(t); res(); } }, 50);
    });
    return true;
  }

  // The move id for a 13-count vector of cards (0 / unknown => kPASS).
  function moveIdForCards(counts) {
    return cardsToId.get(counts.join(',')) ?? kPASS;
  }

  // The 13-count card vector a move id plays (all zeros for pass).
  function cardsForMove(id) { return tbl.moves[id].cards; }

  // Number of legal non-pass plays our hand has against lastMoveId.
  function legalMoveCount(handCounts, lastMoveId) {
    return tbl.computeLegalMoves(handCounts, lastMoveId).filter((m) => m !== kPASS).length;
  }

  // Ask the worker for the CPU's move. Resolves to a move id (kPASS => pass).
  //   req = { hand:[13], discard:[13], oppCount, lastMove, history:[ids],
  //           myPts, oppPts }   (myPts/oppPts = CPU/human series points)
  function selectMove(req) {
    const reqId = ++seq;
    return new Promise((resolve, reject) => {
      pending.set(reqId, { resolve, reject });
      worker.postMessage({ type: 'move', reqId, ...req });
    });
  }

  return {
    init,
    isReady: () => ready,
    moveIdForCards,
    cardsForMove,
    legalMoveCount,
    selectMove,
    kPASS,
  };
})();

window.Big2 = Big2;
export default Big2;
