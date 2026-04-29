use std::collections::HashMap;
use std::sync::OnceLock;

use crate::consts::PASS;
use crate::moves::{Combination, Move, MOVE_TO_CARDS};
use crate::partial_game::PartialGame;
use crate::util::{compute_legal_moves, find_forced_win, opponent_can_respond};

// ---------------------------------------------------------------------------
// Opp-1-card tablebase
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct Opp1Result {
    /// Metric after optimal straight-prefix play (0-12 = rank index, 15 = won).
    pub score: u8,
    /// Move id of the straight to play first, or 0 = use default strategy.
    pub first_move_id: usize,
}

static OPP1_TABLE: OnceLock<HashMap<[u8; 13], Opp1Result>> = OnceLock::new();

/// Load the precomputed opp-1-card tablebase from a binary file.
/// Format: magic "OPP1" (4 bytes), uint32 count, then for each entry:
///   13 bytes hand, 1 byte score, 2 bytes move_id (little-endian u16).
/// Safe to call once at startup; silently a no-op if the file is missing.
pub fn load_tablebase_opp1(path: &str) {
    use std::fs::File;
    use std::io::Read;

    let mut map = HashMap::new();
    if let Ok(mut f) = File::open(path) {
        let mut magic = [0u8; 4];
        if f.read_exact(&mut magic).is_ok()
            && &magic == b"OPP1"
        {
            let mut count_bytes = [0u8; 4];
            if f.read_exact(&mut count_bytes).is_ok() {
                let count = u32::from_le_bytes(count_bytes) as usize;
                for _ in 0..count {
                    let mut hand = [0u8; 13];
                    let mut score_byte = [0u8; 1];
                    let mut lo = [0u8; 1];
                    let mut hi = [0u8; 1];
                    if f.read_exact(&mut hand).is_err()
                        || f.read_exact(&mut score_byte).is_err()
                        || f.read_exact(&mut lo).is_err()
                        || f.read_exact(&mut hi).is_err()
                    {
                        break;
                    }
                    let move_id = lo[0] as usize | ((hi[0] as usize) << 8);
                    map.insert(hand, Opp1Result { score: score_byte[0], first_move_id: move_id });
                }
            }
        }
    }
    // Ignore error if already set (called from multiple threads in theory).
    let _ = OPP1_TABLE.set(map);
}

/// Look up a hand in the opp-1-card table.
pub fn lookup_opp1(hand: &[u8; 13]) -> Opp1Result {
    OPP1_TABLE
        .get()
        .and_then(|t| t.get(hand).copied())
        .unwrap_or_default()
}

/// Default strategy when opp has 1 card and no table entry:
/// bomb (min id) → triple → double → longest straight → lowest single.
pub fn opp1_default_strategy_move(hand: &[u8; 13]) -> Option<usize> {
    let legal = compute_legal_moves(hand, Move::PASS);

    // Bomb
    if let Some(&mid) = legal.iter().filter(|&&m| m != PASS).find(|&&m| {
        Move::decode(m).combination == Combination::Bomb
    }) {
        return Some(mid);
    }
    // Triple
    if let Some(&mid) = legal.iter().filter(|&&m| m != PASS).find(|&&m| {
        Move::decode(m).combination == Combination::Triple
    }) {
        return Some(mid);
    }
    // Double
    if let Some(&mid) = legal.iter().filter(|&&m| m != PASS).find(|&&m| {
        Move::decode(m).combination == Combination::Double
    }) {
        return Some(mid);
    }
    // Longest straight (ties: lowest id)
    let mut best_straight: Option<usize> = None;
    let mut best_len = 0u8;
    for &mid in &legal {
        if mid == PASS { continue; }
        let m = Move::decode(mid);
        if m.combination.is_straight() {
            let len = m.num_cards();
            if len > best_len || (len == best_len && best_straight.map_or(true, |b| mid < b)) {
                best_len = len;
                best_straight = Some(mid);
            }
        }
    }
    if best_straight.is_some() { return best_straight; }

    // Lowest single
    let mut best_single: Option<usize> = None;
    let mut best_rank = u8::MAX;
    for &mid in &legal {
        if mid == PASS { continue; }
        let m = Move::decode(mid);
        if m.combination == Combination::Single && m.rank < best_rank {
            best_rank = m.rank;
            best_single = Some(mid);
        }
    }
    best_single
}

// ---------------------------------------------------------------------------
// Tablebase peek (same logic as Player::select_move's tablebase check)
// ---------------------------------------------------------------------------

pub struct TablebasePeekResult {
    pub mv: Option<Move>,
    /// -1 = no tablebase case, 1 = forced-win sequence, 2 = opp has 1 card.
    pub tb_case: i8,
    pub tb_forced_seq_len: usize,
    pub tb_opp1_table_straight: bool,
}

impl TablebasePeekResult {
    fn none() -> Self {
        TablebasePeekResult { mv: None, tb_case: -1, tb_forced_seq_len: 0, tb_opp1_table_straight: false }
    }
}

/// Determine the tablebase move (if any) for the given `PartialGame` state.
pub fn peek_tablebase_move(game: &PartialGame) -> TablebasePeekResult {
    let hand_size = game.num_cards();
    let legal = game.legal_moves();

    // Case 0: a move that empties our hand — immediate win.
    for &mid in &legal {
        if mid == PASS { continue; }
        if MOVE_TO_CARDS[mid][13] == hand_size {
            return TablebasePeekResult {
                mv: Some(Move::decode(mid)),
                tb_case: -1, // not tagged; always the last turn
                tb_forced_seq_len: 0,
                tb_opp1_table_straight: false,
            };
        }
    }

    if game.last_move().combination == Combination::Pass {
        // Case 2: opponent has exactly 1 card.
        if game.opponent_hand_size() == 1 {
            // If all legal moves are singles, play the highest one.
            let non_pass: Vec<usize> = legal.iter().copied().filter(|&m| m != PASS).collect();
            if non_pass.iter().all(|&m| Move::decode(m).combination == Combination::Single) {
                if let Some(&best) = non_pass.iter().max_by_key(|&&m| Move::decode(m).rank) {
                    return TablebasePeekResult {
                        mv: Some(Move::decode(best)),
                        tb_case: 2,
                        tb_forced_seq_len: 0,
                        tb_opp1_table_straight: false,
                    };
                }
            }

            // Look up the precomputed table.
            let hand = game.player_hand();
            let opp1 = lookup_opp1(&hand);
            if opp1.first_move_id != 0 {
                if legal.contains(&opp1.first_move_id) {
                    return TablebasePeekResult {
                        mv: Some(Move::decode(opp1.first_move_id)),
                        tb_case: 2,
                        tb_forced_seq_len: 0,
                        tb_opp1_table_straight: true,
                    };
                }
            }
            if let Some(def) = opp1_default_strategy_move(&hand) {
                return TablebasePeekResult {
                    mv: Some(Move::decode(def)),
                    tb_case: 2,
                    tb_forced_seq_len: 0,
                    tb_opp1_table_straight: false,
                };
            }
        }

        // Case 1: forced-win sequence.
        let hand = game.player_hand();
        let discard = game.discard_pile();
        if let Some(seq) = find_forced_win(&hand, &discard, game.opponent_hand_size()) {
            return TablebasePeekResult {
                mv: Some(Move::decode(seq[0])),
                tb_case: 1,
                tb_forced_seq_len: seq.len(),
                tb_opp1_table_straight: false,
            };
        }
    }

    TablebasePeekResult::none()
}

// Suppress unused-import warning when tablebase file isn't present.
#[allow(dead_code)]
fn _use_opponent_can_respond() {
    let _ = opponent_can_respond;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opp1_default_strategy_singles_only() {
        // Hand: just a single A and a single 2.
        let mut hand = [0u8; 13];
        hand[11] = 1; // A
        hand[12] = 1; // 2
        // Default strategy: no bomb, no triple, no double, no straight → lowest single.
        // Lowest single is A (rank 14), not 2 (rank 15).
        let mid = opp1_default_strategy_move(&hand);
        assert!(mid.is_some());
        let m = Move::decode(mid.unwrap());
        assert_eq!(m.combination, Combination::Single);
        assert_eq!(m.rank, 14); // A is rank 14, which is lower rank value than 15 (2)
    }

    #[test]
    fn test_peek_tablebase_immediate_win() {
        use crate::game::Game;

        // Build a game state where one player has exactly one single card left.
        // Construct manually: player 0 has only one 3.
        let mut hand0 = [0u8; 13];
        hand0[0] = 1; // one 3
        let mut hand1 = [0u8; 13];
        hand1[1] = 1; // opponent has one 4

        let g = Game::from_state(hand0, hand1, [0; 13], Move::PASS, 0);
        let pg = crate::partial_game::PartialGame::from_game(&g, 0);

        let result = peek_tablebase_move(&pg);
        assert!(result.mv.is_some());
        let mv = result.mv.unwrap();
        // Should play the single 3 (hand-emptying move).
        assert_eq!(mv.combination, Combination::Single);
        assert_eq!(mv.rank, 3);
    }
}
