/**
 * Web Worker: ONNX Runtime Web (WASM) + plain-tree MCTS for the Big2 AZ player.
 * Runs off the main thread so the UI stays responsive.
 *
 * model.onnx I/O (from nn/export_az_seq_onnx.py):
 *   inputs : tokens [1,T] int64,  hand [1,48] f32,  oppmax [1,48] f32,  sides [1,5] f32
 *   outputs: value [1,1],  policy [1,138],  behavior [1,458],  qa [1,458]
 *   sides = [oppSize/16, ourSize/16, ownerToMove?1:0, 0, 0]   (series pts = 0)
 *
 * Message in : { type:'move', hand:[13], discard:[13], oppCount, lastMove, history:[ids], sims? }
 * Message out: { type:'move', moveId }   |   { type:'ready' }   |   { type:'error', error }
 */
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs';
import { MoveTable, kPASS, kUs } from './engine.js';
import { Search } from './mcts.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
ort.env.wasm.numThreads = 1;

const MODEL_URL = new URL('./model.onnx', import.meta.url).href;
const MOVES_URL = new URL('./az_moves.json', import.meta.url).href;
const SERIES_URL = new URL('./series_v_gen5.json', import.meta.url).href;
const DEFAULT_SIMS = 100;
const PTS_NORM = 50.0;   // kPtsNorm / kSeriesTarget (matches the training pipeline)

let tbl = null;
let session = null;
let seriesTable = null;   // 50x50 win-prob v[a][b], leader-perspective

async function init() {
  const [movesJson, seriesJson] = await Promise.all([
    (await fetch(MOVES_URL)).json(),
    (await fetch(SERIES_URL)).json(),
  ]);
  tbl = new MoveTable(movesJson);
  seriesTable = seriesJson;
  session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
}
const ready = init();

// Run one leaf evaluation through ONNX. feat from Search.selectLeaf().
async function evalLeaf(feat) {
  const T = feat.tokens.length;
  const tokens = BigInt64Array.from(feat.tokens.map((x) => BigInt(x)));
  const hand = new Float32Array(48);
  const oppmax = new Float32Array(48);
  tbl.encodeExact(feat.ourHand, hand, 0);
  tbl.encodeThermo(feat.oppMax, oppmax, 0);
  const sides = Float32Array.from([
    feat.oppSize / 16, feat.ourSize / 16, feat.ownerToMove ? 1 : 0,
    (feat.myPts | 0) / PTS_NORM, (feat.oppPts | 0) / PTS_NORM,
  ]);
  const feeds = {
    tokens: new ort.Tensor('int64', tokens, [1, T]),
    hand: new ort.Tensor('float32', hand, [1, 48]),
    oppmax: new ort.Tensor('float32', oppmax, [1, 48]),
    sides: new ort.Tensor('float32', sides, [1, 5]),
  };
  const out = await session.run(feeds);
  return {
    value: out.value.data[0],
    policy: out.policy.data,     // Float32Array[138]
    behavior: out.behavior.data, // Float32Array[458]
    qa: out.qa.data,             // Float32Array[458]
  };
}

async function chooseMove(msg) {
  const root = {
    ourHand: msg.hand.slice(),
    discard: msg.discard.slice(),
    oppSize: msg.oppCount,
    lastMove: msg.lastMove,
    side: kUs,
    myPts: msg.myPts | 0,    // CPU (root player) series points
    oppPts: msg.oppPts | 0,  // human opponent series points
  };
  const search = new Search(tbl, root, msg.history || [], {
    sims: msg.sims || DEFAULT_SIMS, training: false, seriesTable,
  });
  const sims = msg.sims || DEFAULT_SIMS;
  for (let i = 0; i < sims; ++i) {
    const req = search.selectLeaf();
    if (req) search.applyEval(await evalLeaf(req.feat));
  }
  search.finalize();
  return search.bestMove();
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    await ready;
    if (msg.type === 'ping') { self.postMessage({ type: 'ready' }); return; }
    if (msg.type === 'move') {
      const moveId = await chooseMove(msg);
      self.postMessage({ type: 'move', moveId, reqId: msg.reqId });
    }
  } catch (e) {
    self.postMessage({ type: 'error', error: String(e && e.stack || e) });
  }
};

ready.then(() => self.postMessage({ type: 'ready' }))
     .catch((e) => self.postMessage({ type: 'error', error: String(e) }));
