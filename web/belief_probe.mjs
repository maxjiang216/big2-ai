// belief_probe.mjs — how much of the opponent's hidden hand can be inferred from
// their play, and how much of that does the search's opponent model capture?
//
// Games: the az_seq champion plays itself, SAMPLING from its policy head (no
// MCTS) so the generating policy pi is known exactly. Bayes with pi as the
// likelihood is then the best possible belief against this opponent — an upper
// bound on what any hand inference can extract.
//
// At random probe positions (player P to move) we compare beliefs over the
// opponent's current hand:
//   uniform  — card counting only: every k-subset of the unseen cards equally
//              likely (the 16 undealt cards make this far from trivial).
//   bayes    — uniform proposals importance-weighted by
//              prod_s pi(opp's actual move at s | hypothetical hand at s).
//   behavior — the net's behavior head (the opponent model az_search expands);
//              no explicit hand belief, so it is scored on response prediction.
//
//   cd web && node belief_probe.mjs --games=50 --probes=4 --samples=256 \
//       --seed=1 --model=./model.onnx --out=probe_1.jsonl
//   node belief_probe.mjs --report probe_*.jsonl
//
// The shipped model.onnx has a fixed batch of 1 (works, ~8x slower). For a
// batched copy, rewrite input/output dim 0 to a symbol 'B' with the onnx
// python package; outputs are bit-identical.
import { readFileSync, appendFileSync } from 'fs';
import { MoveTable, kPASS, handSize } from './engine.js';

const argv = process.argv.slice(2);
if (argv[0] === '--report') { report(argv.slice(1)); process.exit(0); }

const ort = (await import('onnxruntime-node')).default;
const args = Object.fromEntries(argv.map((a) => { const [k, v] = a.replace(/^--/, '').split('='); return [k, v ?? '1']; }));
const NGAMES = +(args.games ?? 50);
const PROBES = +(args.probes ?? 4);   // probe positions per game
const K = +(args.samples ?? 256);     // uniform hand samples per probe
const NMOVES = +(args.moves ?? 4);    // P's candidate moves scored per probe
const SEED = +(args.seed ?? 1);
const OUT = args.out ?? null;

const tbl = new MoveTable(JSON.parse(readFileSync('./az_moves.json', 'utf8')));
const session = await ort.InferenceSession.create(args.model ?? './model.onnx',
  { intraOpNumThreads: +(args.threads ?? 1) });

function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), a | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t >>> 0) / 4294967296); };
}
const rng = mulberry32(SEED * 0x9e3779b1);

// ---------------------------------------------------------------- net eval --
// feats: [{ ourHand, oppMax, oppSize, ourSize, ownerToMove }] sharing `tokens`.
let maxBatch = 512;
async function evalBatch(tokens, feats) {
  const out = [];
  for (let off = 0; off < feats.length; off += maxBatch) {
    const chunk = feats.slice(off, off + maxBatch);
    let res;
    try { res = await runChunk(tokens, chunk); } catch (e) {
      if (maxBatch === 1) throw e;
      maxBatch = 1; off -= maxBatch; continue;  // fixed-batch model: fall back
    }
    out.push(...res);
  }
  return out;
}
async function runChunk(tokens, feats) {
  const B = feats.length, T = Math.max(tokens.length, 1);
  const tok = new BigInt64Array(B * T);
  for (let b = 0; b < B; ++b) for (let i = 0; i < tokens.length; ++i) tok[b * T + i] = BigInt(tokens[i]);
  const hand = new Float32Array(B * 48), om = new Float32Array(B * 48), sides = new Float32Array(B * 5);
  feats.forEach((f, b) => {
    tbl.encodeExact(f.ourHand, hand, b * 48);
    tbl.encodeThermo(f.oppMax, om, b * 48);
    sides.set([f.oppSize / 16, f.ourSize / 16, f.ownerToMove ? 1 : 0, 0, 0], b * 5);
  });
  const o = await session.run({
    tokens: new ort.Tensor('int64', tok, [B, T]),
    hand: new ort.Tensor('float32', hand, [B, 48]),
    oppmax: new ort.Tensor('float32', om, [B, 48]),
    sides: new ort.Tensor('float32', sides, [B, 5]),
  });
  const P = o.policy.dims[1], H = o.behavior.dims[1];
  return feats.map((_, b) => ({ policy: o.policy.data.subarray(b * P, (b + 1) * P),
                                behavior: o.behavior.data.subarray(b * H, (b + 1) * H) }));
}

// Owner-to-move feature for `hand` facing `oppHand`'s size.
const ownerFeat = (hand, oppSize, discard) => ({
  ourHand: hand, oppMax: tbl.oppMaxCounts(hand, discard), oppSize,
  ourSize: handSize(hand), ownerToMove: true,
});
// Log-softmax of the composed policy over `legal`, evaluated at `target`.
function policyLogProb(policy, legal, target) {
  let mx = -1e30;
  const ls = legal.map((m) => { const l = tbl.composedLogit(m, policy); mx = Math.max(mx, l); return l; });
  let s = 0; for (const l of ls) s += Math.exp(l - mx);
  return tbl.composedLogit(target, policy) - mx - Math.log(s);
}
const sub = (a, b) => a.map((x, r) => x - b[r]);
const add = (a, b) => a.map((x, r) => x + b[r]);

// ------------------------------------------------------- self-play games --
function deal() {
  const deck = [];
  for (let r = 0; r < 13; ++r) for (let k = 0; k < tbl.maxDeck[r]; ++k) deck.push(r);
  for (let i = deck.length - 1; i > 0; --i) { const j = Math.floor(rng() * (i + 1)); [deck[i], deck[j]] = [deck[j], deck[i]]; }
  const h = [new Array(13).fill(0), new Array(13).fill(0)];
  for (let i = 0; i < 32; ++i) h[i < 16 ? 0 : 1][deck[i]]++;
  return h;
}

async function playGame() {
  const hands = deal();
  const discard = new Array(13).fill(0);
  let turn = hands[0][0] > 0 && hands[1][0] === 0 ? 0 : hands[1][0] > 0 && hands[0][0] === 0 ? 1 : 0;
  let last = kPASS;
  const history = [], plies = [];
  for (let t = 0; t < 400; ++t) {
    const legal = tbl.computeLegalMoves(hands[turn], last);
    let mv = legal[0];
    if (legal.length > 1) {
      const [e] = await evalBatch(history, [ownerFeat(hands[turn], handSize(hands[1 - turn]), discard)]);
      const lp = legal.map((m) => policyLogProb(e.policy, legal, m));
      let u = rng(); mv = legal[legal.length - 1];
      for (let i = 0; i < legal.length; ++i) { u -= Math.exp(lp[i]); if (u <= 0) { mv = legal[i]; break; } }
    }
    plies.push({ seat: turn, move: mv, last, hands: [hands[0].slice(), hands[1].slice()], discard: discard.slice() });
    history.push(mv);
    if (mv === kPASS) { last = kPASS; turn = 1 - turn; continue; }
    const cost = tbl.moves[mv].cards;
    for (let r = 0; r < 13; ++r) { hands[turn][r] -= cost[r]; discard[r] += cost[r]; }
    if (handSize(hands[turn]) === 0) break;
    last = mv; turn = 1 - turn;
  }
  return { plies, history };
}

// ------------------------------------------------------------- probe math --
const lgam = (() => { const c = [0]; for (let i = 1; i <= 64; ++i) c.push(c[i - 1] + Math.log(i)); return (n) => c[n]; })();
const lC = (n, k) => (k < 0 || k > n ? -Infinity : lgam(n) - lgam(k) - lgam(n - k));
function sampleHand(u, k) {
  const pool = [];
  for (let r = 0; r < 13; ++r) for (let i = 0; i < u[r]; ++i) pool.push(r);
  const h = new Array(13).fill(0);
  for (let i = 0; i < k; ++i) { const j = i + Math.floor(rng() * (pool.length - i)); [pool[i], pool[j]] = [pool[j], pool[i]]; h[pool[i]]++; }
  return h;
}
const uniformLogProb = (u, U, k, h) => h.reduce((s, c, r) => s + lC(u[r], c), 0) - lC(U, k);
const logMeanExp = (xs) => { const m = Math.max(...xs); return m + Math.log(xs.reduce((s, x) => s + Math.exp(x - m), 0) / xs.length); };
const canBeat = (h, m) => tbl.computeLegalMoves(h, m).some((x) => x !== kPASS);

async function probe(game, t) {
  const { plies, history } = game;
  const pt = plies[t], P = pt.seat, O = 1 - P;
  const Hp = pt.hands[P], Ho = pt.hands[O], disc = pt.discard;
  const u = Hp.map((c, r) => tbl.maxDeck[r] - c - disc[r]);
  const U = u.reduce((a, b) => a + b, 0), k = handSize(Ho);

  // hyps[0] = true hand, hyps[1..K] = uniform samples.
  const hyps = [Ho, ...Array.from({ length: K }, () => sampleHand(u, k))];
  const llPlay = new Float64Array(K + 1), llPass = new Float64Array(K + 1);

  // Replay every opponent decision before t under each hypothesis.
  let after = new Array(13).fill(0);  // cards O played in plies s..t-1
  let nOppDecisions = 0;
  for (let s = t - 1; s >= 0; --s) {
    const ps = plies[s];
    if (ps.seat !== O) continue;
    if (ps.move !== kPASS) after = add(after, tbl.moves[ps.move].cards);
    const need = [], feats = [];
    hyps.forEach((h, j) => {
      const hs = add(h, after);
      const legal = tbl.computeLegalMoves(hs, ps.last);
      if (legal.length > 1) { need.push([j, legal]); feats.push(ownerFeat(hs, handSize(ps.hands[P]), ps.discard)); }
    });
    if (need.length === 0) continue;
    ++nOppDecisions;
    const evs = await evalBatch(history.slice(0, s), feats);
    need.forEach(([j, legal], i) => {
      const lp = policyLogProb(evs[i].policy, legal, ps.move);
      if (ps.move === kPASS) llPass[j] += lp; else llPlay[j] += lp;
    });
  }
  const ll = Array.from(llPlay, (x, j) => x + llPass[j]);
  const llS = ll.slice(1);

  // Importance weights over the samples (true hand excluded: no leakage).
  const mx = Math.max(...llS);
  const w = llS.map((x) => Math.exp(x - mx));
  const wsum = w.reduce((a, b) => a + b, 0);
  const wn = w.map((x) => x / wsum);
  const ess = 1 / wn.reduce((s, x) => s + x * x, 0);

  // Information about the full hand: log posterior/prior of the true hand.
  const ig = (ll[0] - logMeanExp(llS)) / Math.LN2;
  const igPlay = (llPlay[0] - logMeanExp(Array.from(llPlay).slice(1))) / Math.LN2;
  const igPass = (llPass[0] - logMeanExp(Array.from(llPass).slice(1))) / Math.LN2;
  const igQuarter = (ll[0] - logMeanExp(llS.slice(0, Math.max(1, K >> 2)))) / Math.LN2;
  const h0 = -hyps.slice(1).reduce((s, h) => s + uniformLogProb(u, U, k, h), 0) / K / Math.LN2;

  // Per-rank marginal log-loss of the true count (bits, summed over ranks).
  let rankUni = 0, rankBayes = 0;
  for (let r = 0; r < 13; ++r) {
    const c = Ho[r];
    const pu = Math.exp(lC(u[r], c) + lC(U - u[r], k - c) - lC(U, k));
    let pb = 0; hyps.slice(1).forEach((h, j) => { if (h[r] === c) pb += wn[j]; });
    pb = 0.99 * pb + 0.01 * pu;  // smoothing so a missed count is not -inf
    rankUni -= Math.log2(pu); rankBayes -= Math.log2(pb);
  }

  // Decision-relevant: for some of P's moves, can / will the opponent respond?
  const moves = [];
  const cand = tbl.computeLegalMoves(Hp, pt.last).filter((m) => m !== kPASS && tbl.moves[m].total < handSize(Hp));
  for (let i = cand.length - 1; i > 0; --i) { const j = Math.floor(rng() * (i + 1)); [cand[i], cand[j]] = [cand[j], cand[i]]; }
  const toks = history.slice(0, t).concat([cand[0]]);
  for (const m of cand.slice(0, NMOVES)) {
    toks[t] = m;
    const cost = tbl.moves[m].cards;
    const Hp2 = sub(Hp, cost), disc2 = add(disc, cost);
    const feats = [], legals = [];
    hyps.forEach((h) => {
      const legal = tbl.computeLegalMoves(h, m);
      legals.push(legal);
      feats.push(ownerFeat(h, handSize(Hp2), disc2));
    });
    const behFeat = { ourHand: Hp2, oppMax: tbl.oppMaxCounts(Hp2, disc2), oppSize: k,
                      ourSize: handSize(Hp2), ownerToMove: false };
    const evs = await evalBatch(toks, [...feats, behFeat]);
    const resp = hyps.map((h, j) => legals[j].length === 1 ? 0 : 1 - Math.exp(policyLogProb(evs[j].policy, legals[j], kPASS)));
    const beat = hyps.map((h) => canBeat(h, m) ? 1 : 0);
    // Behavior head: masked softmax over the publicly-possible responses.
    const poss = tbl.computePossibleMoves(Hp2, disc2, k, m, false);
    const beh = evs[K + 1].behavior;
    let bmx = -1e30; for (const x of poss) bmx = Math.max(bmx, beh[tbl.moves[x].oppHead]);
    let bs = 0, bp = 0;
    for (const x of poss) { const e = Math.exp(beh[tbl.moves[x].oppHead] - bmx); bs += e; if (x === kPASS) bp = e; }
    const mean = (xs) => xs.slice(1).reduce((a, b) => a + b, 0) / K;
    const wmean = (xs) => xs.slice(1).reduce((a, x, j) => a + wn[j] * x, 0);
    moves.push({
      combo: tbl.moves[m].combo,
      beatTrue: beat[0], beatUni: mean(beat), beatBayes: wmean(beat),
      respTrue: resp[0], respUni: mean(resp), respBayes: wmean(resp), respBeh: 1 - bp / bs,
    });
  }

  return { t, k, lead: pt.last === kPASS, nOppDecisions, h0, ig, igPlay, igPass, igQuarter,
           ess, rankUni, rankBayes, moves };
}

// -------------------------------------------------------------------- main --
const t0 = performance.now();
let nProbes = 0;
for (let g = 0; g < NGAMES; ++g) {
  const game = await playGame();
  // Probe positions: P to move, opponent has made >= 1 decision.
  const cands = [];
  game.plies.forEach((p, t) => { if (game.plies.slice(0, t).some((q) => q.seat !== p.seat)) cands.push(t); });
  for (let i = cands.length - 1; i > 0; --i) { const j = Math.floor(rng() * (i + 1)); [cands[i], cands[j]] = [cands[j], cands[i]]; }
  for (const t of cands.slice(0, PROBES)) {
    const r = await probe(game, t);
    ++nProbes;
    if (OUT) appendFileSync(OUT, JSON.stringify(r) + '\n');
  }
  process.stderr.write(`game ${g + 1}/${NGAMES}  probes ${nProbes}  ${((performance.now() - t0) / 1000).toFixed(0)}s\r`);
}
process.stderr.write('\n');

// ------------------------------------------------------------------ report --
function report(files) {
  const rows = files.flatMap((f) => readFileSync(f, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l)));
  const avg = (xs) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
  const xent = (p, q) => { q = Math.min(1 - 1e-4, Math.max(1e-4, q)); return -(p * Math.log2(q) + (1 - p) * Math.log2(1 - q)); };
  const bands = [['all', () => true], ['opp 13-16', (r) => r.k >= 13], ['opp 9-12', (r) => r.k >= 9 && r.k <= 12],
                 ['opp 5-8', (r) => r.k >= 5 && r.k <= 8], ['opp 1-4', (r) => r.k <= 4]];
  console.log(`probes: ${rows.length}\n`);
  console.log('Hand-level information (bits)');
  console.log('band        n     prior H  bayes IG  (plays/passes)  IG@K/4   resolved  rank-LL uni -> bayes   med ESS');
  for (const [name, f] of bands) {
    const rs = rows.filter(f); if (!rs.length) continue;
    const ess = rs.map((r) => r.ess).sort((a, b) => a - b)[rs.length >> 1];
    console.log(`${name.padEnd(10)} ${String(rs.length).padStart(4)}  ${avg(rs.map((r) => r.h0)).toFixed(1).padStart(8)}  ${avg(rs.map((r) => r.ig)).toFixed(2).padStart(8)}  (${avg(rs.map((r) => r.igPlay)).toFixed(2)}/${avg(rs.map((r) => r.igPass)).toFixed(2)})`.padEnd(54) +
      `${avg(rs.map((r) => r.igQuarter)).toFixed(2).padStart(6)}   ${(100 * avg(rs.map((r) => r.ig)) / avg(rs.map((r) => r.h0))).toFixed(1).padStart(5)}%   ${avg(rs.map((r) => r.rankUni)).toFixed(2)} -> ${avg(rs.map((r) => r.rankBayes)).toFixed(2)}`.padEnd(24) + `   ${ess.toFixed(0).padStart(5)}`);
  }
  console.log('\nResponse prediction for P\'s candidate moves (cross-entropy, bits; lower is better)');
  console.log('band        n     can-beat: uni   bayes  | will-respond: uni   bayes   behavior-head');
  for (const [name, f] of bands) {
    const ms = rows.filter(f).flatMap((r) => r.moves); if (!ms.length) continue;
    console.log(`${name.padEnd(10)} ${String(ms.length).padStart(5)}      ${avg(ms.map((m) => xent(m.beatTrue, m.beatUni))).toFixed(3)}   ${avg(ms.map((m) => xent(m.beatTrue, m.beatBayes))).toFixed(3)}  |            ${avg(ms.map((m) => xent(m.respTrue, m.respUni))).toFixed(3)}   ${avg(ms.map((m) => xent(m.respTrue, m.respBayes))).toFixed(3)}   ${avg(ms.map((m) => xent(m.respTrue, m.respBeh))).toFixed(3)}`);
  }
  // Floor for will-respond: entropy of the true response probability itself.
  const ms = rows.flatMap((r) => r.moves);
  console.log(`\nwill-respond floor (policy's own randomness, all): ${avg(ms.map((m) => xent(m.respTrue, m.respTrue))).toFixed(3)} bits`);
}
