'use strict';
// Plain-tree MCTS — faithful port of src/players/az_search/az_search.cpp for the
// browser. NO transpositions (invalid with move histories): every node has one
// parent. Single game, sims=100, training=false (deterministic opp representative
// = max-prob member), series=null (win_value=1, loss_value=0).
//
// Async eval flow (ONNX runs in the worker): the search is request/response, so
// no blocking needed —
//     for (let s=0; s<sims; ++s) {
//       const req = search.selectLeaf();
//       if (req) search.applyEval(await evalLeaf(req.feat));
//     }
//     search.finalize();           // eval-free forced-WIN proof
//     const move = search.bestMove();
//
// v1 OMISSIONS vs C++ (endgame strength refinements; core search identical):
//   - opp1 in-tree tablebase (opp1_series_move): series-shed line, skipped.
//   - forced_expand_root NN-valued extension: needs synchronous eval inside the
//     root finalize; skipped (post-budget root refinement only).
// The hand-emptying auto-win and the eval-free root forced-WIN proof ARE ported.

import {
  kUs, kPASS, handSize, azTransition, normalizeForcedPass, groupOppMoves,
  seriesValueAfterWin,
} from './engine.js';
import { solveLead, solveResponse, opp1Belief } from './opp1.js';

const COMBO_BOMB = 5, COMBO_FH = 4;

// Group key for hierarchical selection (mirrors az_search.cpp group_key):
// trick type, with full houses / bombs also split by rank.
function groupKey(tbl, moveId) {
  const mv = tbl.moves[moveId];
  if (mv.combo === COMBO_FH || mv.combo === COMBO_BOMB) return mv.combo * 32 + mv.rank;
  return mv.combo;
}

function newNode(st) {
  return {
    st, expanded: false, terminal: false, forcedWinMove: -1,
    nnValue: 0.5, value: 0.5, N: 0, edges: [], groups: [],
  };
}

const edgeN = (e) => (e.child ? e.child.N : 0);

export class Search {
  constructor(tbl, rootState, history, cfg) {
    this.tbl = tbl;
    this.cPuct = cfg.cPuct ?? 1.5;
    this.sims = cfg.sims ?? 100;
    this.training = cfg.training ?? false;
    // Series win-prob table v[a][b] (50x50) or null. When set, terminal/forced
    // values become V(resulting series state) instead of 1/0 (port of C++
    // cfg_.series). Root player's points come from rootState.myPts/oppPts.
    this.seriesTable = cfg.seriesTable ?? null;
    this.history = history.slice();   // game move ids up to (excl.) the root
    this.path = [];
    this.pathTokens = [];
    this.pending = null;
    this.root = newNode(rootState);
    this.finalizeTerminal(this.root);
  }

  // No series table => legacy single-game objective (win=1, loss=0).
  // With a table, value = P(root player wins the series) after this game ends.
  winValue(s) {
    if (!this.seriesTable) return 1.0;
    // Root player wins; opponent is the loser holding s.oppSize cards.
    return seriesValueAfterWin(this.seriesTable, s.myPts, s.oppPts, s.oppSize);
  }

  lossValue(s) {
    if (!this.seriesTable) return 0.0;
    // Opponent wins; root player is the loser holding hand_size(ourHand) cards.
    return 1.0 - seriesValueAfterWin(this.seriesTable, s.oppPts, s.myPts,
                                     handSize(s.ourHand));
  }

  finalizeTerminal(n) {
    if (handSize(n.st.ourHand) === 0) { n.terminal = true; n.expanded = true; n.nnValue = n.value = this.winValue(n.st); }
    else if (n.st.oppSize === 0) { n.terminal = true; n.expanded = true; n.nnValue = n.value = this.lossValue(n.st); }
  }

  expandPlayer(n, policy138) {
    const tbl = this.tbl;
    const ourSize = handSize(n.st.ourHand);
    const legal = tbl.computeLegalMoves(n.st.ourHand, n.st.lastMove);

    // Auto-win: any move that empties our hand wins outright. Prefer largest, lowest id.
    let win = -1, winCards = -1;
    for (const m of legal) {
      if (m === kPASS) continue;
      const c = tbl.moves[m].total;
      if (c === ourSize && c > winCards) { win = m; winCards = c; }
    }
    if (win !== -1) {
      n.terminal = true; n.forcedWinMove = win;
      // All hand-emptying moves leave opp at the same size => identical value.
      n.nnValue = n.value = this.winValue(n.st);
      return;
    }

    // Opp-has-1-card endgame: pin the exact, series-aware solver line. A 1-card
    // opponent can't beat any multi-card combo; the solver also handles straights
    // (brute-force over straight packings) and the response position (point 7),
    // and is belief-free wherever a k-dominating line exists. See web/opp1.js
    // (port of src/core/opp1_solver). Replaces the straight-blind opp1SeriesMove.
    if (n.st.oppSize === 1) {
      const belief = opp1Belief(tbl, n.st.ourHand, n.st.discard);
      const sol = (n.st.lastMove === kPASS)
        ? solveLead(tbl, n.st.ourHand, belief, this.seriesTable, n.st.myPts, n.st.oppPts)
        : solveResponse(tbl, n.st.ourHand, n.st.lastMove, belief, this.seriesTable, n.st.myPts, n.st.oppPts);
      const mv = sol.moves.length ? sol.moves[0] : kPASS;
      if (mv !== kPASS) {
        n.edges.push({ moveId: mv, prior: 1.0, child: null, fusedPass: false, moveValue: 0.5 });
        this.buildGroups(n);
        return;
      }
    }

    // Factored composed-logit masked softmax over the legal concrete moves.
    let maxl = -1e30;
    for (const m of legal) maxl = Math.max(maxl, tbl.composedLogit(m, policy138));
    let sum = 0;
    for (const m of legal) {
      const e = Math.exp(tbl.composedLogit(m, policy138) - maxl);
      n.edges.push({ moveId: m, prior: e, child: null, fusedPass: false, moveValue: 0.5 });
      sum += e;
    }
    if (sum > 0) for (const e of n.edges) e.prior /= sum;
    this.buildGroups(n);
  }

  expandOpp(n, qa, behavior) {
    const tbl = this.tbl;
    const poss = tbl.computePossibleMoves(n.st.ourHand, n.st.discard, n.st.oppSize,
                                          n.st.lastMove, false);
    // Behavior priors (masked softmax over the opp head logits of the plausible set).
    let maxl = -1e30;
    for (const m of poss) maxl = Math.max(maxl, behavior[tbl.moves[m].oppHead]);
    let sum = 0;
    const probs = [];
    for (const m of poss) { const e = Math.exp(behavior[tbl.moves[m].oppHead] - maxl); probs.push(e); sum += e; }
    if (sum > 0) for (let i = 0; i < probs.length; ++i) probs[i] /= sum;

    // Collapse response-equivalent classes; one edge per class with summed prob.
    // Play-time (training=false): representative = max-probability member.
    const classes = groupOppMoves(tbl, n.st.ourHand, poss, probs);
    for (const c of classes) {
      if (c.moves.length === 0) continue;
      let psum = 0; for (const p of c.probs) psum += p;
      let rep = c.moves[0], bestP = -1;
      for (let i = 0; i < c.moves.length; ++i) if (c.probs[i] > bestP) { bestP = c.probs[i]; rep = c.moves[i]; }
      n.edges.push({ moveId: rep, prior: psum, child: null, fusedPass: false, moveValue: 0.5 });
    }
    this.buildGroups(n);

    // Control-variate leaf value: prior-weighted mean of per-move q_a baselines.
    let v = 0;
    for (const e of n.edges) {
      e.moveValue = qa[tbl.moves[e.moveId].oppHead];
      v += e.prior * e.moveValue;
    }
    n.nnValue = n.value = n.edges.length === 0 ? 0.5 : v;
  }

  buildGroups(n) {
    const tbl = this.tbl;
    n.groups = [];
    if (n.st.side !== kUs) {
      // Opponent node: flat 2-level grouping by trick type (FH/bombs by rank).
      for (let i = 0; i < n.edges.length; ++i) {
        const k = groupKey(tbl, n.edges[i].moveId);
        let g = n.groups.find((eg) => eg.key === k);
        if (!g) { g = { key: k, idx: [], sub: [], priorSum: 0 }; n.groups.push(g); }
        g.idx.push(i); g.priorSum += n.edges[i].prior;
      }
      return;
    }
    // Player node: factored 3-level tree — family -> subgroup -> leaf.
    for (let i = 0; i < n.edges.length; ++i) {
      const fam = tbl.moves[n.edges[i].moveId].family;
      const sk = tbl.moves[n.edges[i].moveId].subKey;
      let g = n.groups.find((eg) => eg.key === fam);
      if (!g) { g = { key: fam, idx: [], sub: [], priorSum: 0 }; n.groups.push(g); }
      g.priorSum += n.edges[i].prior;
      let sg = g.sub.find((s) => s.key === sk);
      if (!sg) { sg = { key: sk, idx: [], sub: [], priorSum: 0 }; g.sub.push(sg); }
      sg.priorSum += n.edges[i].prior;
      sg.idx.push(i);
    }
  }

  // Opponent: flat 2-level most-undersampled selection (prob / (visits+1)).
  selectOpp(n) {
    let bg = null, bgScore = -1e30;
    for (const g of n.groups) {
      let Ng = 0; for (const i of g.idx) Ng += edgeN(n.edges[i]);
      const score = g.priorSum / (Ng + 1);
      if (score > bgScore) { bgScore = score; bg = g; }
    }
    let best = null, bestScore = -1e30;
    for (const i of bg.idx) {
      const e = n.edges[i];
      const score = e.prior / (edgeN(e) + 1);
      if (score > bestScore) { bestScore = score; best = e; }
    }
    return best;
  }

  // Player: hierarchical PUCT down the factored tree (conditional priors, max-Q).
  selectPlayer(n) {
    const c = this.cPuct;
    const childQ = (e) => (e.child && e.child.expanded) ? e.child.value : n.value;
    // Level 1: family.
    let Ntot = 0; for (const e of n.edges) Ntot += edgeN(e);
    const sqTot = Math.sqrt(Ntot + 1.0);
    let bg = null, bgs = -1e30;
    for (const g of n.groups) {
      let Ng = 0, Qg = -1.0;
      for (const sg of g.sub) for (const i of sg.idx) { Ng += edgeN(n.edges[i]); Qg = Math.max(Qg, childQ(n.edges[i])); }
      const score = Qg + c * g.priorSum * sqTot / (1.0 + Ng);
      if (score > bgs) { bgs = score; bg = g; }
    }
    // Level 2: subgroup (conditional prior within family).
    let NgF = 0; for (const sg of bg.sub) for (const i of sg.idx) NgF += edgeN(n.edges[i]);
    const sqG = Math.sqrt(NgF + 1.0);
    const invBg = bg.priorSum > 0 ? 1.0 / bg.priorSum : 0;
    let bsg = null, bsgs = -1e30;
    for (const sg of bg.sub) {
      let Ns = 0, Qs = -1.0;
      for (const i of sg.idx) { Ns += edgeN(n.edges[i]); Qs = Math.max(Qs, childQ(n.edges[i])); }
      const pcond = sg.priorSum * invBg;
      const score = Qs + c * pcond * sqG / (1.0 + Ns);
      if (score > bsgs) { bsgs = score; bsg = sg; }
    }
    // Level 3: leaf edge (conditional prior within subgroup).
    let Ns = 0; for (const i of bsg.idx) Ns += edgeN(n.edges[i]);
    const sqS = Math.sqrt(Ns + 1.0);
    const invSg = bsg.priorSum > 0 ? 1.0 / bsg.priorSum : 0;
    let best = null, bes = -1e30;
    for (const i of bsg.idx) {
      const e = n.edges[i];
      const pcond = e.prior * invSg;
      const score = childQ(e) + c * pcond * sqS / (1.0 + edgeN(e));
      if (score > bes) { bes = score; best = e; }
    }
    return best;
  }

  selectEdge(n) {
    if (n.edges.length === 0) return null;
    if (n.groups.length === 0) this.buildGroups(n);
    return (n.st.side === kUs) ? this.selectPlayer(n) : this.selectOpp(n);
  }

  resolveChild(parent, e) {
    if (e.child) return e.child;
    const raw = azTransition(this.tbl, parent.st, e.moveId);
    const cs = azTransition(this.tbl, parent.st, e.moveId); // independent copy to normalize
    normalizeForcedPass(this.tbl, cs);
    e.fusedPass = (cs.side !== raw.side);
    const c = newNode(cs);
    this.finalizeTerminal(c);
    e.child = c;
    return c;
  }

  // Returns { needsEval:true, feat } or null (terminal / backed-up, no eval).
  selectLeaf() {
    this.path = [];
    this.pathTokens = [];
    let cur = this.root;
    for (;;) {
      this.path.push(cur);
      if (cur.terminal) { this.backup(); return null; }
      if (!cur.expanded) {
        this.pending = cur;
        const st = cur.st;
        return {
          needsEval: true,
          feat: {
            ourHand: st.ourHand,
            oppMax: this.tbl.oppMaxCounts(st.ourHand, st.discard),
            oppSize: st.oppSize,
            ourSize: handSize(st.ourHand),
            ownerToMove: st.side === kUs,
            myPts: st.myPts | 0,   // root player's series points (constant/game)
            oppPts: st.oppPts | 0,
            tokens: this.history.concat(this.pathTokens), // full move-id sequence
          },
        };
      }
      const e = this.selectEdge(cur);
      if (!e) { this.backup(); return null; }
      cur = this.resolveChild(cur, e);
      this.pathTokens.push(e.moveId);
      if (e.fusedPass) this.pathTokens.push(kPASS);
    }
  }

  recomputeValue(n) {
    if (n.terminal || !n.expanded) return;
    if (n.st.side === kUs) {
      let best = -1.0, any = false;
      for (const e of n.edges) if (e.child && e.child.expanded) { best = Math.max(best, e.child.value); any = true; }
      n.value = any ? best : n.nnValue;
    } else {
      if (n.edges.length === 0) { n.value = n.nnValue; return; }
      let v = 0;
      for (const e of n.edges) v += e.prior * ((e.child && e.child.expanded) ? e.child.value : e.moveValue);
      n.value = v;
    }
  }

  backup() {
    for (const n of this.path) n.N += 1;
    for (let i = this.path.length - 1; i >= 0; --i) this.recomputeValue(this.path[i]);
  }

  // netEval = { value, policy:[138], behavior:[458], qa:[458] }
  applyEval(netEval) {
    const n = this.pending;
    n.expanded = true;
    if (n.st.side === kUs) {
      n.nnValue = netEval.value; n.value = netEval.value;
      this.expandPlayer(n, netEval.policy);
    } else {
      this.expandOpp(n, netEval.qa, netEval.behavior);
    }
    this.backup();
    this.pending = null;
  }

  // ---- forced-win proof (eval-free), ported from find_forced_win ----------
  oppUpperBound(hand, discard) {
    const ub = new Array(13);
    for (let r = 0; r < 13; ++r) ub[r] = this.tbl.maxDeck[r] - hand[r] - discard[r];
    return ub;
  }

  findForcedWin(hand, discard, oppCount) {
    const ub = this.oppUpperBound(hand, discard);
    const inner = (h, d) => {
      const hs = handSize(h);
      const legal = this.tbl.computeLegalMoves(h, kPASS)
        .filter((m) => m !== kPASS)
        .sort((a, b) => {
          const ca = this.tbl.moves[a].total, cb = this.tbl.moves[b].total;
          return ca !== cb ? cb - ca : a - b;
        });
      for (const mid of legal) {
        const cards = this.tbl.moves[mid].total;
        if (cards === hs) return [mid];
        if (!this.tbl.opponentCanRespond(mid, ub, oppCount)) {
          const cost = this.tbl.moves[mid].cards;
          const nh = h.slice(), nd = d.slice();
          for (let r = 0; r < 13; ++r) { nh[r] -= cost[r]; nd[r] += cost[r]; }
          const sub = inner(nh, nd);
          if (sub) return [mid, ...sub];
        }
      }
      return null;
    };
    return inner(hand.slice(), discard.slice());
  }

  applyRootForcedWin() {
    const n = this.root;
    if (n.terminal) return;
    if (n.st.lastMove !== kPASS) return;        // need the initiative (lead)
    const seq = this.findForcedWin(n.st.ourHand, n.st.discard, n.st.oppSize);
    if (seq) { n.forcedWinMove = seq[0]; n.value = this.winValue(n.st); }
  }

  finalize() { this.applyRootForcedWin(); }

  bestMove() {
    const n = this.root;
    if (n.forcedWinMove !== -1) return n.forcedWinMove;
    let best = -1, bestN = -1, bestPrior = -1;
    for (const e of n.edges) {
      const cn = e.child ? e.child.N : 0;
      if (cn > bestN || (cn === bestN && e.prior > bestPrior)) { bestN = cn; bestPrior = e.prior; best = e.moveId; }
    }
    return best;
  }
}
