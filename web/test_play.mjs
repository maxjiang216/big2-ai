// Node end-to-end strength harness: JS-MCTS player (real model.onnx via
// onnxruntime-node) vs a baseline, playing full Big2 games on the JS engine.
// Validates engine.js + mcts.js + the ONNX net together. If MCTS dominates a
// random opponent (~>85%), the whole JS stack works.
//
//   cd web && node test_play.mjs [games] [sims] [opponent=random]
import { readFileSync } from 'fs';
import ort from 'onnxruntime-node';
import { MoveTable, kPASS, kUs, handSize } from './engine.js';
import { Search } from './mcts.js';

const NGAMES = parseInt(process.argv[2] || '20', 10);
const SIMS = parseInt(process.argv[3] || '64', 10);
const OPP = process.argv[4] || 'random';

const tbl = new MoveTable(JSON.parse(readFileSync('./az_moves.json', 'utf8')));
const session = await ort.InferenceSession.create('./model.onnx');

function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), a | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t >>> 0) / 4294967296); };
}
// Mix a counter into a well-distributed 32-bit seed (avoids consecutive-seed
// correlation in mulberry32).
function splitmix32(n) {
  let z = (n + 0x9e3779b9) | 0;
  z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
  z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
  return (z ^ (z >>> 15)) >>> 0;
}

// Deal: 48-card multiset (maxDeck per rank) -> two 16-card hands; 16 unused.
function deal(rng) {
  const deck = [];
  for (let r = 0; r < 13; ++r) for (let k = 0; k < tbl.maxDeck[r]; ++k) deck.push(r);
  for (let i = deck.length - 1; i > 0; --i) { const j = Math.floor(rng() * (i + 1)); [deck[i], deck[j]] = [deck[j], deck[i]]; }
  const h0 = new Array(13).fill(0), h1 = new Array(13).fill(0);
  for (let i = 0; i < 16; ++i) h0[deck[i]]++;
  for (let i = 16; i < 32; ++i) h1[deck[i]]++;
  return [h0, h1];
}

async function evalLeaf(feat) {
  const T = feat.tokens.length;
  const tokens = BigInt64Array.from(feat.tokens.map((x) => BigInt(x)));
  const hand = new Float32Array(48), oppmax = new Float32Array(48);
  tbl.encodeExact(feat.ourHand, hand, 0);
  tbl.encodeThermo(feat.oppMax, oppmax, 0);
  const sides = Float32Array.from([feat.oppSize / 16, feat.ourSize / 16, feat.ownerToMove ? 1 : 0, 0, 0]);
  const o = await session.run({
    tokens: new ort.Tensor('int64', tokens, [1, T]),
    hand: new ort.Tensor('float32', hand, [1, 48]),
    oppmax: new ort.Tensor('float32', oppmax, [1, 48]),
    sides: new ort.Tensor('float32', sides, [1, 5]),
  });
  return { value: o.value.data[0], policy: o.policy.data, behavior: o.behavior.data, qa: o.qa.data };
}

async function mctsMove(myHand, discard, oppSize, lastMove, history) {
  const root = { ourHand: myHand.slice(), discard: discard.slice(), oppSize, lastMove, side: kUs };
  const s = new Search(tbl, root, history, { sims: SIMS, training: false });
  for (let i = 0; i < SIMS; ++i) { const req = s.selectLeaf(); if (req) s.applyEval(await evalLeaf(req.feat)); }
  s.finalize();
  return s.bestMove();
}

function randomMove(myHand, lastMove, rng) {
  const legal = tbl.computeLegalMoves(myHand, lastMove); // lead set has no PASS
  return legal[Math.floor(rng() * legal.length)];
}

// Greedy: among non-pass legal moves, pick the one whose resulting hand scores
// best lexicographically (win_now, bombs, -cards, then rank counts 2,A,K..4).
// Never voluntarily passes. Port of greedy_hand_eval / greedy_best.
function greedyScore(hand) {
  let nCards = 0, nBombs = 0;
  for (let i = 0; i < 13; ++i) { nCards += hand[i]; if ((i < 11 && hand[i] === 4) || (i === 11 && hand[i] === 3)) nBombs++; }
  // [win_now, bombs, -cards, 2,A,K,Q,J,10,9,8,7,6,5,4] (rank idx 12,11,10,...,1)
  return [nCards === 0 ? 1 : 0, nBombs, -nCards, hand[12], hand[11], hand[10], hand[9], hand[8], hand[7], hand[6], hand[5], hand[4], hand[3], hand[2], hand[1]];
}
function lexGt(a, b) { for (let i = 0; i < a.length; ++i) { if (a[i] !== b[i]) return a[i] > b[i]; } return false; }
function greedyMove(myHand, lastMove) {
  const legal = tbl.computeLegalMoves(myHand, lastMove);
  let best = -1, bestScore = null;
  for (const m of legal) {
    if (m === kPASS) continue;
    const cost = tbl.moves[m].cards;
    const h = myHand.slice();
    for (let r = 0; r < 13; ++r) h[r] -= cost[r];
    const sc = greedyScore(h);
    if (best === -1 || lexGt(sc, bestScore)) { best = m; bestScore = sc; }
  }
  return best === -1 ? kPASS : best;
}

// Deal-determined first seat (proxy for the 3♠ holder): seat with the 3 (rank
// idx 0); ties / neither -> seat 0. Fixed across both halves of a pair so the
// first-player advantage flips with the seat swap and cancels.
function firstSeatFor(hands) {
  if (hands[0][0] > 0 && hands[1][0] === 0) return 0;
  if (hands[1][0] > 0 && hands[0][0] === 0) return 1;
  return 0;
}

// Play one game with given hands. players[seat] in {'mcts','greedy','random'}.
async function playGame(players, hands, rng, firstPlayer) {
  const discard = new Array(13).fill(0);
  let turn = firstPlayer, lastMove = kPASS;
  const history = []; // public move-id sequence
  for (let ply = 0; ply < 400; ++ply) {
    const me = players[turn];
    let mv;
    if (me === 'mcts') {
      mv = await mctsMove(hands[turn], discard, handSize(hands[1 - turn]), lastMove, history);
    } else if (me === 'greedy') {
      mv = greedyMove(hands[turn], lastMove);
    } else {
      mv = randomMove(hands[turn], lastMove, rng);
    }
    history.push(mv);
    if (mv === kPASS) { lastMove = kPASS; turn = 1 - turn; continue; } // opp led last -> opp leads
    const cost = tbl.moves[mv].cards;
    for (let r = 0; r < 13; ++r) { hands[turn][r] -= cost[r]; discard[r] += cost[r]; }
    if (handSize(hands[turn]) === 0) return turn;
    lastMove = mv; turn = 1 - turn;
  }
  return -1; // safety
}

// Paired eval: each deal played both seat assignments, first seat fixed by the
// deal -> cancels card luck AND first-player advantage (matches eval_az_match).
let mctsWins = 0, total = 0, sweeps = 0;
for (let p = 0; p < NGAMES; ++p) {
  const hands = deal(mulberry32(splitmix32(1000 + p)));  // one deal per pair
  const fs = firstSeatFor(hands);
  // Half A: mcts=seat0. Half B: mcts=seat1 (swap controllers, same hands/firstseat).
  const wA = await playGame(['mcts', OPP], [hands[0].slice(), hands[1].slice()], null, fs);
  const wB = await playGame([OPP, 'mcts'], [hands[0].slice(), hands[1].slice()], null, fs);
  const mctsA = (wA === 0) ? 1 : 0;   // mcts is seat 0 in A
  const mctsB = (wB === 1) ? 1 : 0;   // mcts is seat 1 in B
  mctsWins += mctsA + mctsB; total += 2;
  if (mctsA + mctsB === 2) sweeps++;
  process.stdout.write(`pair ${p + 1}/${NGAMES}: A=${mctsA} B=${mctsB}  running ${(mctsWins / total).toFixed(3)}\r`);
}
console.log(`\nMCTS(${SIMS} sims) vs ${OPP} (paired): ${mctsWins}/${total} = ${(mctsWins / total).toFixed(3)}  (mcts swept ${sweeps}/${NGAMES} pairs)`);
