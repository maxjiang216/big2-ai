// Node test for web/opp1.js: cross-check the JS solver against an independent
// brute-force optimum (mirrors test/test_opp1_solver.cpp) + specific cases.
//   node web/test_opp1.mjs
import { readFileSync } from 'node:fs';
import { MoveTable, kPASS } from './engine.js';
import { enumeratePlans, kBelow, solveLead, solveResponse } from './opp1.js';

const tbl = new MoveTable(JSON.parse(readFileSync(new URL('./az_moves.json', import.meta.url))));
const CAP = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1];
let fails = 0;
const ok = (c, m) => { if (!c) { console.error('FAIL:', m); fails++; } };

function handOf(pairs) { const h = new Array(13).fill(0); for (let i = 0; i < pairs.length; i += 2) h[pairs[i]] = pairs[i + 1]; return h; }
function sum(h) { let s = 0; for (const x of h) s += x; return s; }

// Brute-force optimum: our remaining cards on optimal loss (-1 = win), opp rank X.
const memoMap = new Map();
function bruteBest(hand, X) {
  const total = sum(hand);
  if (total === 0) return -1;
  const key = X + '|' + hand.join(',');
  if (memoMap.has(key)) return memoMap.get(key);
  let best = total;
  for (const id of tbl.computeLegalMoves(hand, kPASS)) {
    if (id === kPASS) continue;
    const mv = tbl.moves[id];
    const remTotal = total - mv.total;
    if (remTotal === 0) { best = -1; break; }
    let cand;
    if (mv.combo === 1 && mv.rank < X) { cand = remTotal; }
    else {
      const rem = hand.slice();
      for (let r = 0; r < 13; ++r) rem[r] -= mv.cards[r];
      cand = bruteBest(rem, X);
    }
    if (cand === -1) { best = -1; break; }
    best = Math.min(best, cand);
  }
  memoMap.set(key, best);
  return best;
}

// 1) Solver enumeration achieves the brute optimum for every opp rank.
let rng = 12345;
const rand = () => (rng = (rng * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
for (let trial = 0; trial < 4000; ++trial) {
  const n = 2 + Math.floor(rand() * 8);  // 2..9
  const pool = [];
  for (let r = 0; r < 13; ++r) for (let k = 0; k < CAP[r]; ++k) pool.push(r);
  for (let i = pool.length - 1; i > 0; --i) { const j = Math.floor(rand() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
  const h = new Array(13).fill(0);
  for (let i = 0; i < n; ++i) h[pool[i]]++;
  const plans = enumeratePlans(tbl, h);
  for (let X = 3; X <= 15; ++X) {
    let solverBest = 1 << 30;
    for (const p of plans) { const k = kBelow(p, X); solverBest = Math.min(solverBest, k <= 1 ? -1 : k - 1); }
    memoMap.clear();
    ok(solverBest === bruteBest(h, X), `brute mismatch hand=${h} X=${X} solver=${solverBest} bf=${bruteBest(h, X)}`);
  }
}

// 2) Bomb-aux second-lowest: four 7s + singles 3,8,K.
{
  const h = handOf([0, 1, 4, 4, 5, 1, 10, 1]);
  const sol = solveLead(tbl, h, new Array(13).fill(1 / 13), null, 0, 0);
  const m = tbl.moves[sol.moves[0]];
  ok(m.combo === 5 && m.rank === 7 && m.aux === 8, `bomb-aux got combo=${m.combo} rank=${m.rank} aux=${m.aux}`);
}
// 3) Straight dominates: singles 3..7 + 9,J.
{
  const h = handOf([0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 6, 1, 8, 1]);
  const sol = solveLead(tbl, h, new Array(13).fill(1 / 13), null, 0, 0);
  ok(sol.byDominance && tbl.moves[sol.moves[0]].combo >= 6, `straight-dom got byDom=${sol.byDominance} combo=${tbl.moves[sol.moves[0]].combo}`);
}
// 4) Response: opp led a single 8 (rank 11), we hold a higher single Q (rank 12,idx9) + junk.
{
  const h = handOf([9, 1, 0, 1]);  // Q (idx9) and 3 (idx0)
  const last = tbl.moveIdFor(1, 11, 0);  // opp played a single rank 11 ('9')... use 8 -> rank 8
  const sol = solveResponse(tbl, h, last, new Array(13).fill(1 / 13), null, 0, 0);
  ok(sol.moves.length === 1, `response returned ${sol.moves.length} moves`);
}

if (fails === 0) console.log('web/opp1.js: ALL PASS (4000 brute-force hands + cases)');
else { console.error(`${fails} failures`); process.exit(1); }
