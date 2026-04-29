pub const LEGAL_MOVES_SIZE: usize = 468;

pub const PASS: usize = 0;
pub const SINGLE_START: usize = 1;
pub const DOUBLE_START: usize = SINGLE_START + 13;
pub const TRIPLE_START: usize = DOUBLE_START + 12;
pub const FULL_HOUSE_START: usize = TRIPLE_START + 11;
pub const BOMB_START: usize = FULL_HOUSE_START + 132;

pub const STRAIGHT5_START: usize = BOMB_START + 156;
pub const STRAIGHT6_START: usize = STRAIGHT5_START + 10;
pub const STRAIGHT7_START: usize = STRAIGHT6_START + 9;
pub const STRAIGHT8_START: usize = STRAIGHT7_START + 8;
pub const STRAIGHT9_START: usize = STRAIGHT8_START + 7;
pub const STRAIGHT10_START: usize = STRAIGHT9_START + 6;
pub const STRAIGHT11_START: usize = STRAIGHT10_START + 5;
pub const STRAIGHT12_START: usize = STRAIGHT11_START + 4;
pub const STRAIGHT13_START: usize = STRAIGHT12_START + 3;

pub const DOUBLESTRAIGHT2_START: usize = STRAIGHT13_START + 1;
pub const DOUBLESTRAIGHT3_START: usize = DOUBLESTRAIGHT2_START + 11;
pub const DOUBLESTRAIGHT4_START: usize = DOUBLESTRAIGHT3_START + 10;
pub const DOUBLESTRAIGHT5_START: usize = DOUBLESTRAIGHT4_START + 9;
pub const DOUBLESTRAIGHT6_START: usize = DOUBLESTRAIGHT5_START + 8;
pub const DOUBLESTRAIGHT7_START: usize = DOUBLESTRAIGHT6_START + 7;

pub const TRIPLESTRAIGHT2_START: usize = DOUBLESTRAIGHT7_START + 6;
pub const TRIPLESTRAIGHT3_START: usize = TRIPLESTRAIGHT2_START + 10;
pub const TRIPLESTRAIGHT4_START: usize = TRIPLESTRAIGHT3_START + 9;
pub const TRIPLESTRAIGHT5_START: usize = TRIPLESTRAIGHT4_START + 8;

pub const DOUBLESTRAIGHT8_START: usize = TRIPLESTRAIGHT5_START + 7;

/// Number of copies of rank `rank_idx` (0-based) in the 48-card deck.
/// rank_idx 11 = Ace (3 copies), rank_idx 12 = Two (1 copy), others = 4.
#[inline]
pub const fn max_copies_in_deck(rank_idx: usize) -> u8 {
    match rank_idx {
        11 => 3,
        12 => 1,
        _ => 4,
    }
}

/// Convert a rank value (3-15) to its 0-based index into the hand array.
/// rank 3 -> 0, rank 4 -> 1, ..., rank 14 (A) -> 11, rank 15 (2-as-high) -> 12.
#[inline]
pub const fn rank_to_idx(rank: u8) -> usize {
    (rank - 3) as usize
}
