use big2_core::Combination;

use crate::game_level::{
    GameLevelFeature, LengthFeature, OutcomeFeature, StartLegalMovesFeature,
    TablebaseHitsFeature, TbCase1Feature, TbCase2Feature, TbForcedSeqLenFeature,
    TbForcedSeqLenSumFeature, TbOpp1TableStraightFeature,
};
use crate::turn_level::{
    BombFeature, HighestRankNotBombFeature, HighestRankWithCountFeature,
    LastMoveCardCountFeature, LastMoveCombinationFeature, LastMoveIsStraightFeature,
    NextPlayerFeature, OnlySingleFeature, OpponentHandSizeFeature, PlayerHandSizeFeature,
    PossibleMovesFeature, PossibleMovesNotBombFeature, RankFeature, RankGeFeature,
    RankLeFeature, TbCaseFeature, TrickRankFeature, TurnLevelFeature, TurnOutcomeFeature,
};

pub enum Feature {
    Game(Box<dyn GameLevelFeature>),
    Turn(Box<dyn TurnLevelFeature>),
}

impl Feature {
    pub fn name(&self) -> &'static str {
        match self {
            Feature::Game(f) => f.name(),
            Feature::Turn(f) => f.name(),
        }
    }
}

/// Returns a `Feature` for the given name, or `None` if unknown.
/// Name strings match the C++ `create_feature` registry exactly.
pub fn create_feature(name: &str) -> Option<Feature> {
    use Feature::*;

    // --- game-level ---
    match name {
        "outcome"                => return Some(Game(Box::new(OutcomeFeature))),
        "length"                 => return Some(Game(Box::new(LengthFeature))),
        "start_legal_moves"      => return Some(Game(Box::new(StartLegalMovesFeature))),
        "tb_hits"                => return Some(Game(Box::new(TablebaseHitsFeature))),
        "tb_case1"               => return Some(Game(Box::new(TbCase1Feature))),
        "tb_case2"               => return Some(Game(Box::new(TbCase2Feature))),
        "tb_forced_seq_len"      => return Some(Game(Box::new(TbForcedSeqLenFeature))),
        "tb_forced_seq_len_sum"  => return Some(Game(Box::new(TbForcedSeqLenSumFeature))),
        "tb_opp1_table_straight" => return Some(Game(Box::new(TbOpp1TableStraightFeature))),
        _ => {}
    }

    // --- turn-level: basic ---
    match name {
        "turn_outcome"       => return Some(Turn(Box::new(TurnOutcomeFeature))),
        "next_player"        => return Some(Turn(Box::new(NextPlayerFeature))),
        "player_hand_size"   => return Some(Turn(Box::new(PlayerHandSizeFeature))),
        "opponent_hand_size" => return Some(Turn(Box::new(OpponentHandSizeFeature))),
        "tb_case"            => return Some(Turn(Box::new(TbCaseFeature))),
        "only_single"        => return Some(Turn(Box::new(OnlySingleFeature))),
        "last_move_card_count"       => return Some(Turn(Box::new(LastMoveCardCountFeature))),
        "n_bombs"                    => return Some(Turn(Box::new(BombFeature))),
        "possible_moves"             => return Some(Turn(Box::new(PossibleMovesFeature))),
        "possible_moves_not_bomb"    => return Some(Turn(Box::new(PossibleMovesNotBombFeature))),
        "trick_rank"                 => return Some(Turn(Box::new(TrickRankFeature))),
        _ => {}
    }

    // --- rank counts n_3 .. n_2 ---
    const RANKS: [(&str, usize, &str); 13] = [
        ("n_3", 0, "n_3"), ("n_4", 1, "n_4"), ("n_5", 2, "n_5"), ("n_6", 3, "n_6"),
        ("n_7", 4, "n_7"), ("n_8", 5, "n_8"), ("n_9", 6, "n_9"), ("n_10", 7, "n_10"),
        ("n_j", 8, "n_j"), ("n_q", 9, "n_q"), ("n_k", 10, "n_k"),
        ("n_a", 11, "n_a"), ("n_2", 12, "n_2"),
    ];
    for (n, idx, feat) in RANKS {
        if name == n {
            return Some(Turn(Box::new(RankFeature { rank: idx, feature_name: feat })));
        }
    }

    // --- n_ge_* ---
    const RANKS_GE: [(&str, usize, &str); 12] = [
        ("n_ge_4",  1, "n_ge_4"),  ("n_ge_5",  2, "n_ge_5"),  ("n_ge_6",  3, "n_ge_6"),
        ("n_ge_7",  4, "n_ge_7"),  ("n_ge_8",  5, "n_ge_8"),  ("n_ge_9",  6, "n_ge_9"),
        ("n_ge_10", 7, "n_ge_10"), ("n_ge_j",  8, "n_ge_j"),  ("n_ge_q",  9, "n_ge_q"),
        ("n_ge_k",  10, "n_ge_k"), ("n_ge_a",  11, "n_ge_a"), ("n_ge_2",  12, "n_ge_2"),
    ];
    for (n, idx, feat) in RANKS_GE {
        if name == n {
            return Some(Turn(Box::new(RankGeFeature { rank: idx, feature_name: feat })));
        }
    }

    // --- n_le_* ---
    const RANKS_LE: [(&str, usize, &str); 12] = [
        ("n_le_3",  0, "n_le_3"),  ("n_le_4",  1, "n_le_4"),  ("n_le_5",  2, "n_le_5"),
        ("n_le_6",  3, "n_le_6"),  ("n_le_7",  4, "n_le_7"),  ("n_le_8",  5, "n_le_8"),
        ("n_le_9",  6, "n_le_9"),  ("n_le_10", 7, "n_le_10"), ("n_le_j",  8, "n_le_j"),
        ("n_le_q",  9, "n_le_q"),  ("n_le_k",  10, "n_le_k"), ("n_le_a",  11, "n_le_a"),
    ];
    for (n, idx, feat) in RANKS_LE {
        if name == n {
            return Some(Turn(Box::new(RankLeFeature { rank: idx, feature_name: feat })));
        }
    }

    // --- highest rank with N copies ---
    match name {
        "highest_single" => return Some(Turn(Box::new(HighestRankWithCountFeature { count: 1, feature_name: "highest_single" }))),
        "highest_double" => return Some(Turn(Box::new(HighestRankWithCountFeature { count: 2, feature_name: "highest_double" }))),
        "highest_triple" => return Some(Turn(Box::new(HighestRankWithCountFeature { count: 3, feature_name: "highest_triple" }))),
        "highest_bomb"   => return Some(Turn(Box::new(HighestRankWithCountFeature { count: 4, feature_name: "highest_bomb"   }))),
        "highest_single_not_bomb" => return Some(Turn(Box::new(HighestRankNotBombFeature { count: 1, feature_name: "highest_single_not_bomb" }))),
        "highest_double_not_bomb" => return Some(Turn(Box::new(HighestRankNotBombFeature { count: 2, feature_name: "highest_double_not_bomb" }))),
        "highest_triple_not_bomb" => return Some(Turn(Box::new(HighestRankNotBombFeature { count: 3, feature_name: "highest_triple_not_bomb" }))),
        _ => {}
    }

    // --- last move combination ---
    match name {
        "last_move_is_pass"       => return Some(Turn(Box::new(LastMoveCombinationFeature { combo: Combination::Pass,      feature_name: "last_move_is_pass"       }))),
        "last_move_is_single"     => return Some(Turn(Box::new(LastMoveCombinationFeature { combo: Combination::Single,    feature_name: "last_move_is_single"     }))),
        "last_move_is_double"     => return Some(Turn(Box::new(LastMoveCombinationFeature { combo: Combination::Double,    feature_name: "last_move_is_double"     }))),
        "last_move_is_triple"     => return Some(Turn(Box::new(LastMoveCombinationFeature { combo: Combination::Triple,    feature_name: "last_move_is_triple"     }))),
        "last_move_is_full_house" => return Some(Turn(Box::new(LastMoveCombinationFeature { combo: Combination::FullHouse, feature_name: "last_move_is_full_house" }))),
        "last_move_is_bomb"       => return Some(Turn(Box::new(LastMoveCombinationFeature { combo: Combination::Bomb,      feature_name: "last_move_is_bomb"       }))),
        "last_move_is_single_straight" => return Some(Turn(Box::new(LastMoveIsStraightFeature { straight_type: 1, feature_name: "last_move_is_single_straight" }))),
        "last_move_is_double_straight" => return Some(Turn(Box::new(LastMoveIsStraightFeature { straight_type: 2, feature_name: "last_move_is_double_straight" }))),
        "last_move_is_triple_straight" => return Some(Turn(Box::new(LastMoveIsStraightFeature { straight_type: 3, feature_name: "last_move_is_triple_straight" }))),
        _ => {}
    }

    eprintln!("Warning: unknown feature '{}', skipping", name);
    None
}
