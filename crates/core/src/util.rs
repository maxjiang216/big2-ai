use crate::consts::*;
use crate::moves::{Combination, Move, MOVE_TO_CARDS};

/// All legal moves from `hand` given that `last_move` was played last.
/// Includes PASS (id 0) when there is an active trick to beat.
pub fn compute_legal_moves(hand: &[u8; 13], last_move: Move) -> Vec<usize> {
    let mut legal = Vec::with_capacity(32);
    let active_trick = last_move.combination != Combination::Pass;
    if active_trick {
        legal.push(PASS);
    }
    'outer: for move_id in SINGLE_START..LEGAL_MOVES_SIZE {
        let mv = Move::decode(move_id);
        if active_trick {
            if mv.combination != Combination::Bomb {
                if mv.combination != last_move.combination || mv.rank <= last_move.rank {
                    continue;
                }
            } else {
                // Bombs can always be played, but must outrank if last was also a bomb.
                if last_move.combination == Combination::Bomb && mv.rank <= last_move.rank {
                    continue;
                }
            }
        }
        let cost = &MOVE_TO_CARDS[move_id];
        for rank in 0..13 {
            if hand[rank] < cost[rank] {
                continue 'outer;
            }
        }
        legal.push(move_id);
    }
    legal
}

/// Moves the opponent might plausibly hold given our hand, discard pile,
/// and opponent card count.  Optionally excludes bombs.
pub fn compute_possible_moves(
    player_hand: &[u8; 13],
    discard: &[u8; 13],
    opp_count: u8,
    last_move: Move,
    exclude_bombs: bool,
) -> Vec<usize> {
    let mut possible = Vec::with_capacity(32);
    let active_trick = last_move.combination != Combination::Pass;
    if active_trick {
        possible.push(PASS);
    }
    'outer: for move_id in SINGLE_START..LEGAL_MOVES_SIZE {
        let mv = Move::decode(move_id);
        if exclude_bombs && mv.combination == Combination::Bomb {
            continue;
        }
        if active_trick {
            if mv.combination != Combination::Bomb {
                if mv.combination != last_move.combination || mv.rank <= last_move.rank {
                    continue;
                }
            } else if last_move.combination == Combination::Bomb && mv.rank <= last_move.rank {
                continue;
            }
        }
        let cost = &MOVE_TO_CARDS[move_id];
        if opp_count < cost[13] {
            continue;
        }
        for rank in 0..13 {
            let unseen = max_copies_in_deck(rank) - player_hand[rank] - discard[rank];
            if unseen < cost[rank] {
                continue 'outer;
            }
        }
        possible.push(move_id);
    }
    possible
}

/// Returns true if all moves in `hand` are only singles (no pairs/triples/etc.).
pub fn hand_is_only_singles(hand: &[u8; 13]) -> bool {
    hand.iter().all(|&c| c <= 1)
}

/// For each non-pass move id, the set of move ids that beat it.
/// Computed once on first call (O(N²) pre-processing).
pub fn get_beating_moves() -> &'static Vec<Vec<usize>> {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<Vec<usize>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = vec![Vec::new(); LEGAL_MOVES_SIZE];
        for mid in SINGLE_START..LEGAL_MOVES_SIZE {
            let m = Move::decode(mid);
            for mid2 in SINGLE_START..LEGAL_MOVES_SIZE {
                let m2 = Move::decode(mid2);
                let beats = if m2.combination == Combination::Bomb {
                    m.combination != Combination::Bomb || m2.rank > m.rank
                } else {
                    m2.combination == m.combination && m2.rank > m.rank
                };
                if beats {
                    t[mid].push(mid2);
                }
            }
        }
        t
    })
}

/// True if the opponent might hold at least one response to `move_id`.
/// Only checks bare bombs (no kicker) as the minimal-card response.
pub fn opponent_can_respond(
    move_id: usize,
    hand: &[u8; 13],
    discard: &[u8; 13],
    opp_count: u8,
) -> bool {
    for &mid2 in &get_beating_moves()[move_id] {
        let m2 = Move::decode(mid2);
        // For bombs: only check bare (no-kicker) variant.
        if m2.combination == Combination::Bomb && m2.auxiliary != 0 {
            continue;
        }
        let cost = &MOVE_TO_CARDS[mid2];
        if opp_count < cost[13] {
            continue;
        }
        let mut feasible = true;
        for rank in 0..13 {
            let unseen = max_copies_in_deck(rank) - hand[rank] - discard[rank];
            if unseen < cost[rank] {
                feasible = false;
                break;
            }
        }
        if feasible {
            return true;
        }
    }
    false
}

/// From a lead position, find a sequence of moves that guarantees emptying our hand.
/// Returns the first complete winning sequence found (DFS, most-cards-first order),
/// or None if no forced win exists.
pub fn find_forced_win(
    hand: &[u8; 13],
    discard: &[u8; 13],
    opp_count: u8,
) -> Option<Vec<usize>> {
    let hand_size: u8 = hand.iter().sum();

    let mut legal = compute_legal_moves(hand, Move::PASS);
    legal.retain(|&m| m != PASS);
    // Sort: most cards first, then lower move_id for ties.
    legal.sort_unstable_by(|&a, &b| {
        let ca = MOVE_TO_CARDS[a][13];
        let cb = MOVE_TO_CARDS[b][13];
        cb.cmp(&ca).then(a.cmp(&b))
    });

    for mid in legal {
        let cards = MOVE_TO_CARDS[mid][13];

        if cards == hand_size {
            return Some(vec![mid]);
        }

        if !opponent_can_respond(mid, hand, discard, opp_count) {
            let cost = &MOVE_TO_CARDS[mid];
            let mut new_hand = *hand;
            let mut new_discard = *discard;
            for rank in 0..13 {
                new_hand[rank] -= cost[rank];
                new_discard[rank] += cost[rank];
            }
            if let Some(mut rest) = find_forced_win(&new_hand, &new_discard, opp_count) {
                rest.insert(0, mid);
                return Some(rest);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_legal_moves_new_trick() {
        // Start of game: all rank-3s in hand, last move is pass.
        let mut hand = [0u8; 13];
        hand[0] = 4; // four 3s
        let legal = compute_legal_moves(&hand, Move::PASS);
        // Should not include PASS (no active trick).
        assert!(!legal.contains(&PASS));
        // Should include single 3 (id 1), double 3 (id 14), triple 3 (id 26).
        assert!(legal.contains(&1));
        assert!(legal.contains(&14));
        assert!(legal.contains(&26));
    }

    #[test]
    fn test_compute_legal_moves_response() {
        // Response to a single 3: must beat it or pass.
        let mut hand = [0u8; 13];
        hand[1] = 1; // one 4
        hand[11] = 1; // one A
        let last = Move::new(Combination::Single, 3, 0);
        let legal = compute_legal_moves(&hand, last);
        assert!(legal.contains(&PASS));
        // Single 4 (id=2) and single A (id=12) should be legal.
        assert!(legal.contains(&2));
        assert!(legal.contains(&12));
        // Single 3 (id=1) must NOT be legal (not higher).
        assert!(!legal.contains(&1));
    }

    #[test]
    fn test_find_forced_win_unit() {
        let mut hand = [0u8; 13];
        hand[11] = 1; // one A
        hand[12] = 1; // the only 2

        let mut discard = [0u8; 13];
        for r in 0..11 {
            discard[r] = 4; // all ranks 3-K fully discarded
        }
        discard[11] = 1; // one A discarded

        let seq = find_forced_win(&hand, &discard, 1);
        assert!(seq.is_some());
        let seq = seq.unwrap();
        assert_eq!(seq.len(), 2);
        let m0 = Move::decode(seq[0]);
        let m1 = Move::decode(seq[1]);
        assert_eq!(m0.combination, Combination::Single);
        assert_eq!(m1.combination, Combination::Single);
        let ranks: std::collections::HashSet<u8> = [m0.rank, m1.rank].into();
        assert!(ranks.contains(&14)); // A
        assert!(ranks.contains(&15)); // 2
    }

    #[test]
    fn test_find_forced_win_no_sequence() {
        let mut hand = [0u8; 13];
        hand[0] = 1; // single 3
        hand[1] = 1; // single 4

        let discard = [0u8; 13];
        let seq = find_forced_win(&hand, &discard, 10);
        assert!(seq.is_none());
    }
}
