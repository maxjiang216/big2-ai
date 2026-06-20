'use strict';
// Exact opp-1-card endgame solver — JS port of src/core/opp1_solver.cpp.
//
// "We have the lead (or must respond) and the opponent holds exactly 1 card."
// Opp's single can only beat a single we lead, by a strictly higher rank, and
// doing so empties opp -> opp wins. So every multi-card move is unbeatable; the
// cards at risk are the singles we are forced to lead one at a time (the EXPOSED
// set S). For opp card rank X, k = #{s in S : s < X} gives WIN (k<=1, lowest
// exposed single is the free last play) or LOSS by k-1. Only single straights
// and bomb auxiliaries change S, so the brute force is over straight packings.
//
// All combinatorics come from the MoveTable (engine.js / az_moves.json) so this
// cannot drift from the C++ source of truth.

import { kPASS, seriesValueAfterWin } from './engine.js';

const COMBO_SINGLE = 1, COMBO_DOUBLE = 2, COMBO_TRIPLE = 3, COMBO_BOMB = 5;
const COMBO_STRAIGHT5 = 6;  // every combo >= 6 is a straight family

// Belief over opp's hidden single rank: prob[idx] for rank value idx+3, taken
// proportional to the cards opp could still hold (engine oppMaxCounts).
export function opp1Belief(tbl, ourHand, discard) {
  const om = tbl.oppMaxCounts(ourHand, discard);  // [13] counts
  const b = new Array(13).fill(0);
  let tot = 0;
  for (let r = 0; r < 13; ++r) { b[r] = Math.max(0, om[r]); tot += b[r]; }
  for (let r = 0; r < 13; ++r) b[r] = tot > 0 ? b[r] / tot : 0;
  return b;
}

// Build the trivial line for `rem` (13 rank counts): bombs (with the lowest
// loose singles except the global-lowest buried as auxiliaries), triples,
// doubles, then exposed singles highest-first. Returns {moves:[ids], exposed:[ranks asc]}.
function buildTrivial(tbl, rem) {
  const loose = [];        // rank values, ascending (idx 0..12 -> value idx+3)
  for (let r = 0; r < 13; ++r) if (rem[r] === 1) loose.push(r + 3);

  const bombRanks = [];
  for (let r = 0; r <= 10; ++r) if (rem[r] === 4) bombRanks.push(r + 3);
  if (rem[11] === 3) bombRanks.push(14);  // ace bomb

  const nb = bombRanks.length;
  const buriedList = [];
  for (let i = 1; i < loose.length && buriedList.length < nb; ++i) buriedList.push(loose[i]);
  const buried = new Set(buriedList);

  const moves = [];
  let ti = 0;
  for (const br of bombRanks) {
    const aux = ti < buriedList.length ? buriedList[ti++] : 0;
    moves.push(tbl.moveIdFor(COMBO_BOMB, br, aux));
  }
  for (let r = 0; r <= 10; ++r) if (rem[r] === 3) moves.push(tbl.moveIdFor(COMBO_TRIPLE, r + 3, 0));
  for (let r = 0; r < 13; ++r) if (rem[r] === 2) moves.push(tbl.moveIdFor(COMBO_DOUBLE, r + 3, 0));

  const exposed = loose.filter((v) => !buried.has(v));  // ascending
  for (let i = exposed.length - 1; i >= 0; --i) moves.push(tbl.moveIdFor(COMBO_SINGLE, exposed[i], 0));
  return { moves, exposed };
}

function playable(tbl, rem, id) {
  const c = tbl.moves[id].cards;
  for (let r = 0; r < 13; ++r) if (c[r] > rem[r]) return false;
  return true;
}

const NODE_CAP = 200000;

// All candidate lines: trivial plus one per single-straight packing played first.
export function enumeratePlans(tbl, hand) {
  const straights = [];
  for (const id of tbl.computeLegalMoves(hand, kPASS))
    if (tbl.moves[id].combo >= COMBO_STRAIGHT5) straights.push(id);
  straights.sort((a, b) => a - b);

  const out = [];
  const seen = new Set();
  const nodes = { n: 0 };
  const chosen = [];
  const rec = (rem, start) => {
    if (nodes.n++ > NODE_CAP) return;
    const t = buildTrivial(tbl, rem);
    out.push({ moves: chosen.concat(t.moves), exposed: t.exposed });
    for (let i = start; i < straights.length; ++i) {
      const s = straights[i];
      if (!playable(tbl, rem, s)) continue;
      const rem2 = rem.slice();
      const c = tbl.moves[s].cards;
      for (let r = 0; r < 13; ++r) rem2[r] -= c[r];
      const key = rem2.join(',');
      if (seen.has(key)) continue;
      seen.add(key);
      chosen.push(s);
      rec(rem2, i);
      chosen.pop();
    }
  };
  rec(hand.slice(), 0);
  return out;
}

export function kBelow(plan, oppRank) {
  let k = 0;
  for (const s of plan.exposed) if (s < oppRank) ++k;
  return k;
}

// Our series win-probability if opp's card has rank X. table=null -> binary.
export function valueVsRank(plan, oppRank, table, myPts, oppPts) {
  const k = kBelow(plan, oppRank);
  if (k <= 1) return table ? seriesValueAfterWin(table, myPts, oppPts, 1) : 1.0;
  return table ? 1.0 - seriesValueAfterWin(table, oppPts, myPts, k - 1) : 0.0;
}

export function planEv(plan, belief, table, myPts, oppPts) {
  let e = 0;
  for (let r = 0; r < 13; ++r) if (belief[r] > 0) e += belief[r] * valueVsRank(plan, r + 3, table, myPts, oppPts);
  return e;
}

// Lead position. Returns { moves:[ids], byDominance }. If one plan k-dominates
// every other it is played (objective, belief-free); else max-EV under belief.
export function solveLead(tbl, hand, belief, table, myPts, oppPts) {
  const plans = enumeratePlans(tbl, hand);
  if (plans.length === 0) return { moves: [], byDominance: false };

  const kmin = new Array(16).fill(1 << 30);
  for (const p of plans) for (let x = 3; x <= 15; ++x) kmin[x] = Math.min(kmin[x], kBelow(p, x));
  for (const p of plans) {
    let dom = true;
    for (let x = 3; x <= 15; ++x) if (kBelow(p, x) !== kmin[x]) { dom = false; break; }
    if (dom) return { moves: p.moves, byDominance: true };
  }

  let best = -1, bestMoves = plans[0].moves;
  for (const p of plans) {
    const e = planEv(p, belief, table, myPts, oppPts);
    if (e > best) { best = e; bestMoves = p.moves; }
  }
  return { moves: bestMoves, byDominance: false };
}

function winVal(table, myPts, oppPts) {
  return table ? seriesValueAfterWin(table, myPts, oppPts, 1) : 1.0;
}
function lossVal(table, myPts, oppPts, ourCards) {
  return table ? 1.0 - seriesValueAfterWin(table, oppPts, myPts, ourCards) : 0.0;
}

// Response position (point 7): opp just played `lastMoveId` and now holds 1 card.
// A multi-card/bomb response regains the lead (continuation = solveLead); a
// single response of rank sr is beaten iff opp's X > sr. Returns the best first
// response move, or moves:[] when we have no legal response (forced pass = loss).
export function solveResponse(tbl, hand, lastMoveId, belief, table, myPts, oppPts) {
  let ourSize = 0;
  for (let r = 0; r < 13; ++r) ourSize += hand[r];
  const legal = tbl.computeLegalMoves(hand, lastMoveId);

  let bestEv = -1, bestMove = null;
  for (const id of legal) {
    if (id === kPASS) continue;
    const mv = tbl.moves[id];
    const rem = hand.slice();
    for (let r = 0; r < 13; ++r) rem[r] -= mv.cards[r];
    const remTotal = ourSize - mv.total;

    let ev;
    if (remTotal === 0) {
      ev = winVal(table, myPts, oppPts);  // our response empties our hand -> win
    } else {
      const cont = solveLead(tbl, rem, belief, table, myPts, oppPts);
      const cp = { exposed: [] };
      for (const cid of cont.moves) if (tbl.moves[cid].combo === COMBO_SINGLE) cp.exposed.push(tbl.moves[cid].rank);
      cp.exposed.sort((a, b) => a - b);
      const isSingle = mv.combo === COMBO_SINGLE;
      ev = 0;
      for (let xi = 0; xi < 13; ++xi) {
        if (belief[xi] <= 0) continue;
        const X = xi + 3;
        const v = (isSingle && X > mv.rank)
          ? lossVal(table, myPts, oppPts, remTotal)
          : valueVsRank(cp, X, table, myPts, oppPts);
        ev += belief[xi] * v;
      }
    }
    if (ev > bestEv) { bestEv = ev; bestMove = id; }
  }
  return { moves: bestMove === null ? [] : [bestMove], byDominance: false };
}
