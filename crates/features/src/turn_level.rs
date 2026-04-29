use big2_core::{Combination, GameRecord, MOVE_TO_CARDS};

/// Turn-level feature.  `extract_all` returns 2 * num_turns values:
/// for each turn, for each perspective p in [0, 1].
pub trait TurnLevelFeature: Send + Sync {
    fn name(&self) -> &'static str;
    fn extract_all(&self, record: &GameRecord) -> Vec<i32>;
}

// ---------------------------------------------------------------------------
// Basic
// ---------------------------------------------------------------------------

pub struct TurnOutcomeFeature;
impl TurnLevelFeature for TurnOutcomeFeature {
    fn name(&self) -> &'static str { "turn_outcome" }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let winner = record.winner() as i32;
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for _ in &record.turns {
            out.push(winner);        // perspective 0
            out.push(1 - winner);   // perspective 1
        }
        out
    }
}

pub struct NextPlayerFeature;
impl TurnLevelFeature for NextPlayerFeature {
    fn name(&self) -> &'static str { "next_player" }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            let cp = turn.current_player as i32;
            out.push(if cp == 0 { 1 } else { 0 });
            out.push(if cp == 1 { 1 } else { 0 });
        }
        out
    }
}

pub struct PlayerHandSizeFeature;
impl TurnLevelFeature for PlayerHandSizeFeature {
    fn name(&self) -> &'static str { "player_hand_size" }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            out.push(turn.game.hand_size(0) as i32);
            out.push(turn.game.hand_size(1) as i32);
        }
        out
    }
}

pub struct OpponentHandSizeFeature;
impl TurnLevelFeature for OpponentHandSizeFeature {
    fn name(&self) -> &'static str { "opponent_hand_size" }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            out.push(turn.game.hand_size(1) as i32); // opponent of p=0 is p=1
            out.push(turn.game.hand_size(0) as i32); // opponent of p=1 is p=0
        }
        out
    }
}

pub struct TbCaseFeature;
impl TurnLevelFeature for TbCaseFeature {
    fn name(&self) -> &'static str { "tb_case" }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            out.push(turn.tb_case as i32);
            out.push(turn.tb_case as i32);
        }
        out
    }
}

pub struct OnlySingleFeature;
impl TurnLevelFeature for OnlySingleFeature {
    fn name(&self) -> &'static str { "only_single" }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 {
                out.push(if turn.views[p].hand_is_only_singles() { 1 } else { 0 });
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Rank counts
// ---------------------------------------------------------------------------

pub struct RankFeature { pub rank: usize, pub feature_name: &'static str }
impl TurnLevelFeature for RankFeature {
    fn name(&self) -> &'static str { self.feature_name }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let r = self.rank;
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 { out.push(turn.views[p].player_hand()[r] as i32); }
        }
        out
    }
}

pub struct RankGeFeature { pub rank: usize, pub feature_name: &'static str }
impl TurnLevelFeature for RankGeFeature {
    fn name(&self) -> &'static str { self.feature_name }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let r = self.rank;
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 {
                let h = turn.views[p].player_hand();
                let v: u8 = h[r..].iter().sum();
                out.push(v as i32);
            }
        }
        out
    }
}

pub struct RankLeFeature { pub rank: usize, pub feature_name: &'static str }
impl TurnLevelFeature for RankLeFeature {
    fn name(&self) -> &'static str { self.feature_name }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let r = self.rank;
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 {
                let h = turn.views[p].player_hand();
                let v: u8 = h[..=r].iter().sum();
                out.push(v as i32);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Highest rank with N copies
// ---------------------------------------------------------------------------

pub struct HighestRankWithCountFeature { pub count: u8, pub feature_name: &'static str }
impl TurnLevelFeature for HighestRankWithCountFeature {
    fn name(&self) -> &'static str { self.feature_name }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let c = self.count;
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 {
                let h = turn.views[p].player_hand();
                let v = (0..13i32).rev().find(|&i| h[i as usize] >= c).unwrap_or(-1);
                out.push(v);
            }
        }
        out
    }
}

pub struct HighestRankNotBombFeature { pub count: u8, pub feature_name: &'static str }
impl TurnLevelFeature for HighestRankNotBombFeature {
    fn name(&self) -> &'static str { self.feature_name }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let c = self.count;
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 {
                let h = turn.views[p].player_hand();
                let v = (0..13i32).rev().find(|&i| {
                    let ui = i as usize;
                    let is_bomb = (ui == 11 && h[ui] == 3) || (ui != 11 && h[ui] == 4);
                    h[ui] >= c && !is_bomb
                }).unwrap_or(-1);
                out.push(v);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Last move features
// ---------------------------------------------------------------------------

pub struct LastMoveCardCountFeature;
impl TurnLevelFeature for LastMoveCardCountFeature {
    fn name(&self) -> &'static str { "last_move_card_count" }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 {
                let mid = turn.views[p].last_move().encode();
                out.push(MOVE_TO_CARDS[mid][13] as i32);
            }
        }
        out
    }
}

pub struct LastMoveCombinationFeature { pub combo: Combination, pub feature_name: &'static str }
impl TurnLevelFeature for LastMoveCombinationFeature {
    fn name(&self) -> &'static str { self.feature_name }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let c = self.combo;
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 {
                out.push(if turn.views[p].last_move().combination == c { 1 } else { 0 });
            }
        }
        out
    }
}

pub struct LastMoveIsStraightFeature { pub straight_type: u8, pub feature_name: &'static str }
impl TurnLevelFeature for LastMoveIsStraightFeature {
    fn name(&self) -> &'static str { self.feature_name }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let st = self.straight_type;
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 {
                let c = turn.views[p].last_move().combination;
                let v = match st {
                    1 => c.is_straight(),
                    2 => c.is_double_straight(),
                    3 => c.is_triple_straight(),
                    _ => false,
                };
                out.push(if v { 1 } else { 0 });
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

pub struct BombFeature;
impl TurnLevelFeature for BombFeature {
    fn name(&self) -> &'static str { "n_bombs" }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 { out.push(turn.views[p].count_bombs() as i32); }
        }
        out
    }
}

pub struct PossibleMovesFeature;
impl TurnLevelFeature for PossibleMovesFeature {
    fn name(&self) -> &'static str { "possible_moves" }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 {
                let v = if turn.views[p].turn() != 0 {
                    turn.views[p].possible_moves().len() as i32
                } else {
                    0
                };
                out.push(v);
            }
        }
        out
    }
}

pub struct PossibleMovesNotBombFeature;
impl TurnLevelFeature for PossibleMovesNotBombFeature {
    fn name(&self) -> &'static str { "possible_moves_not_bomb" }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 {
                let v = if turn.views[p].turn() != 0 {
                    turn.views[p].possible_moves_not_bomb().len() as i32
                } else {
                    0
                };
                out.push(v);
            }
        }
        out
    }
}

pub struct TrickRankFeature;
impl TurnLevelFeature for TrickRankFeature {
    fn name(&self) -> &'static str { "trick_rank" }
    fn extract_all(&self, record: &GameRecord) -> Vec<i32> {
        let mut out = Vec::with_capacity(record.turns.len() * 2);
        for turn in &record.turns {
            for p in 0..2 {
                out.push(turn.views[p].trick_rank().unwrap_or(0) as i32);
            }
        }
        out
    }
}
