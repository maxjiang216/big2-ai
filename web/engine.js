'use strict';
// Big2 engine for the browser AZ player — a faithful, data-driven port of the
// C++ az_search move logic. ALL combinatorics come from web/az_moves.json (the
// C++ single source of truth dumped by bin/az_moves_gen), so legality /
// possible-moves / grouping / composition cannot drift from the engine.
//
// Rank index convention (matches C++): 0='3' … 10='K', 11='A', 12='2'.
// Sides: kUs = 0 (the AZ player / hand owner), kOpp = 1.
// A move id is an int in [0, 468); kPASS = 0. cards[r] = rank counts of a move.

export const kUs = 0;
export const kOpp = 1;
export const kPASS = 0;

// Loaded once from az_moves.json. Holds the per-move table + global constants.
export class MoveTable {
  constructor(json) {
    this.numMoves = json.num_moves;          // 468
    this.kPASS = json.kPASS;                  // 0
    this.oppHeadDim = json.opp_head_dim;      // 458
    this.playerHeadDim = json.player_head_dim;// 138
    this.maxDeck = json.max_deck;             // [4×11, 3, 1]  (== RANK_MAX)
    this.moves = json.moves;                  // [{id,combo,rank,aux,total,oppHead,family,subKey,cards,path,beating}]
    // Encoding offset table: ENCODING_DIM=48 = sum(maxDeck).
    this.encDim = this.maxDeck.reduce((a, b) => a + b, 0);
    // (combo,rank,aux) -> move id, for constructing specific moves (opp1 shed).
    this.byCRA = new Map();
    for (const m of this.moves) this.byCRA.set(`${m.combo},${m.rank},${m.aux}`, m.id);
  }

  moveIdFor(combo, rank, aux) {
    return this.byCRA.get(`${combo},${rank},${aux}`);
  }

  m(id) { return this.moves[id]; }

  // True iff `hand` (13 rank counts) contains every card a move requires.
  handHasMove(hand, id) {
    const c = this.moves[id].cards;
    for (let r = 0; r < 13; ++r) if (c[r] > hand[r]) return false;
    return true;
  }

  // Our legal moves vs the trick. last == kPASS => lead position.
  //   lead     : { m != PASS : cards[m] ⊆ hand }
  //   response : [PASS] ∪ { m ∈ beating[last] : cards[m] ⊆ hand }
  computeLegalMoves(hand, last) {
    const out = [];
    if (last === kPASS) {
      for (let id = 1; id < this.numMoves; ++id)
        if (this.handHasMove(hand, id)) out.push(id);
      return out;
    }
    out.push(kPASS);
    for (const id of this.moves[last].beating)
      if (this.handHasMove(hand, id)) out.push(id);
    return out;
  }

  // Opponent's publicly-plausible moves, deduced from our hand + discard +
  // opponent card count (their exact hand is hidden).
  //   oppUB[r] = maxDeck[r] - ourHand[r] - discard[r]   (cards opp could hold)
  //   candidate set = (lead ? all non-pass : beating[last])
  //   keep m iff total[m] <= oppCount AND cards[m] ⊆ oppUB  (and bomb filter)
  computePossibleMoves(ourHand, discard, oppCount, last, excludeBombs) {
    const ub = new Array(13);
    for (let r = 0; r < 13; ++r) ub[r] = this.maxDeck[r] - ourHand[r] - discard[r];
    const consider = (id) => {
      const mv = this.moves[id];
      if (excludeBombs && mv.combo === 5 /* kBomb */) return false;
      if (mv.total > oppCount) return false;
      const c = mv.cards;
      for (let r = 0; r < 13; ++r) if (c[r] > ub[r]) return false;
      return true;
    };
    const out = [];
    if (last === kPASS) {
      for (let id = 1; id < this.numMoves; ++id) if (consider(id)) out.push(id);
      return out;
    }
    out.push(kPASS);
    for (const id of this.moves[last].beating) if (consider(id)) out.push(id);
    return out;
  }

  // True if the opponent might still hold a response to `last`, given their
  // per-rank upper bound counts `ub` and card count. Bare bombs only (C++
  // get_beat_entries excludes aux-bomb variants — same beating power, fewer cards).
  opponentCanRespond(last, ub, oppCount) {
    for (const id of this.moves[last].beating) {
      const mv = this.moves[id];
      if (mv.combo === 5 /* kBomb */ && mv.aux !== 0) continue;
      if (mv.total > oppCount) continue;
      const c = mv.cards;
      let ok = true;
      for (let r = 0; r < 13; ++r) if (c[r] > ub[r]) { ok = false; break; }
      if (ok) return true;
    }
    return false;
  }

  // opp_max_counts: per-rank cards the opponent could still hold.
  oppMaxCounts(ourHand, discard) {
    const o = new Array(13);
    for (let r = 0; r < 13; ++r)
      o[r] = Math.max(0, this.maxDeck[r] - ourHand[r] - discard[r]);
    return o;
  }

  // trick_counts: rank counts of the move on the table (zeros for PASS / lead).
  trickCounts(last) {
    if (last === kPASS) return new Array(13).fill(0);
    return this.moves[last].cards.slice();
  }

  // 48-dim exact one-hot: bit i of rank r set iff counts[r] == i+1.
  encodeExact(counts, buf, off = 0) {
    for (let r = 0; r < 13; ++r) {
      const c = counts[r], mx = this.maxDeck[r];
      for (let k = 1; k <= mx; ++k) buf[off++] = (c === k) ? 1 : 0;
    }
    return off;
  }

  // 48-dim thermometer: bit i of rank r set iff counts[r] >= i+1 (capped at max).
  encodeThermo(counts, buf, off = 0) {
    for (let r = 0; r < 13; ++r) {
      const c = Math.min(counts[r], this.maxDeck[r]), mx = this.maxDeck[r];
      for (let k = 1; k <= mx; ++k) buf[off++] = (c >= k) ? 1 : 0;
    }
    return off;
  }

  // Composed player-policy logit = sum of the move's factored path components.
  composedLogit(id, policy138) {
    const p = this.moves[id].path;
    let s = 0;
    for (let i = 0; i < p.length; ++i) s += policy138[p[i]];
    return s;
  }

  // True iff our hand has any straight lead (combo >= kStraight5 == 6). opp1
  // shed is only valid when this is false (a straight beats the 1-card analysis).
  handHasStraightLead(hand) {
    for (const id of this.computeLegalMoves(hand, kPASS))
      if (this.moves[id].combo >= 6) return true;
    return false;
  }

  // Series-optimal lead when WE lead and the opponent holds exactly 1 card and
  // our hand has no straight lead (port of opp1_series_move). Sheds multi-card
  // combos (a 1-card opp can't beat them), smallest loose single as bomb aux,
  // then singles highest-first. Returns a move id (kPASS only if hand empty).
  opp1SeriesMove(hand) {
    // combo ints: kSingle=1 kDouble=2 kTriple=3 kBomb=5. rank/aux use the engine
    // single-rank convention (r+3: idx 11=A->14, idx 12=deuce->15).
    let auxRank = 0;
    for (let r = 0; r < 13; ++r) if (hand[r] === 1) { auxRank = r + 3; break; }
    for (let r = 0; r <= 10; ++r) if (hand[r] === 4) {       // four-of-a-kind bomb
      const bombRank = r + 3;
      const aux = (auxRank !== 0 && auxRank !== bombRank) ? auxRank : 0;
      return this.moveIdFor(5, bombRank, aux);
    }
    if (hand[11] === 3) {                                     // ace bomb (rank 14)
      const aux = (auxRank !== 0 && auxRank !== 14) ? auxRank : 0;
      return this.moveIdFor(5, 14, aux);
    }
    for (let r = 0; r <= 10; ++r) if (hand[r] === 3) return this.moveIdFor(3, r + 3, 0);
    for (let r = 0; r < 13; ++r) if (hand[r] === 2) return this.moveIdFor(2, r + 3, 0);
    for (let r = 12; r >= 0; --r) if (hand[r] === 1) return this.moveIdFor(1, r + 3, 0);
    return kPASS;
  }
}

export function handSize(hand) {
  let s = 0;
  for (let r = 0; r < 13; ++r) s += hand[r];
  return s;
}

// SearchState: { ourHand:[13], discard:[13], oppSize:int, lastMove:int, side:0|1,
//                myPts:int, oppPts:int }
// myPts/oppPts are the ROOT player's (CPU = "us") series points; constant per
// game (the net value head is always read from the root player's perspective).
export const kSeriesTarget = 50;

// Points the game winner scores given the loser's remaining card count.
// 1..12 -> identity; 13 -> 20; 14 -> 30; 15 -> 40; 16 -> 50 (port of series.cpp).
export function pointsForCards(cardsRemaining) {
  if (cardsRemaining <= 0) return 0;
  if (cardsRemaining <= 12) return cardsRemaining;
  return 10 * (cardsRemaining - 11);
}

// P(the game's winner ultimately wins the series), from a 50x50 leader-perspective
// win-prob table v[a][b]. Returns 1.0 if the win ends the series. table===null =>
// legacy single-game objective (caller substitutes 1/0). Port of series.cpp.
export function seriesValueAfterWin(table, winnerPts, loserPts, loserCardsRemaining) {
  const next = winnerPts + pointsForCards(loserCardsRemaining);
  if (next >= kSeriesTarget) return 1.0;
  return table[next][loserPts];
}

export function azTransition(tbl, s, moveId) {
  const ns = {
    ourHand: s.ourHand.slice(),
    discard: s.discard.slice(),
    oppSize: s.oppSize,
    lastMove: s.lastMove,
    side: s.side,
    myPts: s.myPts | 0,
    oppPts: s.oppPts | 0,
  };
  const mover = s.side;
  if (moveId === kPASS) {
    ns.side = 1 - mover;
    ns.lastMove = kPASS;
    return ns;
  }
  const cost = tbl.moves[moveId].cards;
  for (let r = 0; r < 13; ++r) ns.discard[r] += cost[r];
  if (mover === kUs) for (let r = 0; r < 13; ++r) ns.ourHand[r] -= cost[r];
  else ns.oppSize -= tbl.moves[moveId].total;
  ns.lastMove = moveId;
  ns.side = 1 - mover;
  return ns;
}

// Collapse a forced pass (the side to move has no legal non-pass play) into the
// post-pass lead node for the trick winner. Loops for the rare chained case.
export function normalizeForcedPass(tbl, s) {
  for (;;) {
    if (s.lastMove === kPASS) return;                       // lead: must play
    if (handSize(s.ourHand) === 0 || s.oppSize === 0) return; // terminal
    let hasPlay;
    if (s.side === kUs) {
      const legal = tbl.computeLegalMoves(s.ourHand, s.lastMove);
      hasPlay = legal.some((m) => m !== kPASS);
    } else {
      const poss = tbl.computePossibleMoves(s.ourHand, s.discard, s.oppSize,
                                            s.lastMove, false);
      hasPlay = poss.some((m) => m !== kPASS);
    }
    if (hasPlay) return;
    s.side = 1 - s.side;   // forced pass: trick winner (other side) leads
    s.lastMove = kPASS;
  }
}

// Group opponent moves into response-equivalence classes (port of C++
// typed_search::group_opp_moves). PASS = own group; bomb / full-house split by
// rank (collapse aux); other combos merge consecutive ranks when OUR hand has no
// same-combination response strictly between them.
//   moves: int[], probs: float[] (parallel) -> [{moves:[], probs:[]}]
export function groupOppMoves(tbl, ourHand, moves, probs) {
  const out = [];
  if (moves.length === 0) return out;
  const COMBO_BOMB = 5, COMBO_FH = 4;

  const infos = moves.map((mid, i) => {
    if (mid === kPASS) return { combo: -1, rank: -1, idx: i };
    const mv = tbl.moves[mid];
    return { combo: mv.combo, rank: mv.rank, idx: i };
  });
  infos.sort((a, b) => (a.combo !== b.combo) ? a.combo - b.combo : a.rank - b.rank);

  // Sorted unique ranks at which OUR hand has a same-combination response.
  const ourResponseRanks = (combo) => {
    if (combo === COMBO_BOMB || combo === COMBO_FH) return [];
    const set = new Set();
    for (const mid of tbl.computeLegalMoves(ourHand, kPASS)) {
      if (mid === kPASS) continue;
      const mv = tbl.moves[mid];
      if (mv.combo === combo) set.add(mv.rank);
    }
    return Array.from(set).sort((a, b) => a - b);
  };
  const hasResponseInRange = (ranks, Ra, Rb) => {
    for (const r of ranks) if (r > Ra && r <= Rb) return true; // ranks sorted asc
    return false;
  };

  let i = 0;
  while (i < infos.length) {
    const combo = infos[i].combo;
    let end = i;
    while (end < infos.length && infos[end].combo === combo) ++end;

    if (combo === -1) {                       // PASS group
      const g = { moves: [], probs: [] };
      for (let k = i; k < end; ++k) { g.moves.push(moves[infos[k].idx]); g.probs.push(probs[infos[k].idx]); }
      out.push(g);
    } else if (combo === COMBO_BOMB || combo === COMBO_FH) {  // per-rank groups
      let j = i;
      while (j < end) {
        const r = infos[j].rank;
        const g = { moves: [], probs: [] };
        while (j < end && infos[j].rank === r) { g.moves.push(moves[infos[j].idx]); g.probs.push(probs[infos[j].idx]); ++j; }
        out.push(g);
      }
    } else {                                  // merge consecutive ranks
      const ourRanks = ourResponseRanks(combo);
      let j = i, current = null, lastMax = -1;
      while (j < end) {
        const r = infos[j].rank;
        if (current === null || hasResponseInRange(ourRanks, lastMax, r)) {
          current = { moves: [], probs: [] };
          out.push(current);
        }
        lastMax = r;
        while (j < end && infos[j].rank === r) { current.moves.push(moves[infos[j].idx]); current.probs.push(probs[infos[j].idx]); ++j; }
      }
    }
    i = end;
  }
  return out;
}
