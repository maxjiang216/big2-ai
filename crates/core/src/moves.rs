use crate::consts::*;
use std::fmt;

/// All combination types, ordered to match the C++ Combination enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Combination {
    Pass,
    Single,
    Double,
    Triple,
    FullHouse,
    Bomb,
    Straight5,
    Straight6,
    Straight7,
    Straight8,
    Straight9,
    Straight10,
    Straight11,
    Straight12,
    Straight13,
    DoubleStraight2,
    DoubleStraight3,
    DoubleStraight4,
    DoubleStraight5,
    DoubleStraight6,
    DoubleStraight7,
    DoubleStraight8,
    TripleStraight2,
    TripleStraight3,
    TripleStraight4,
    TripleStraight5,
}

impl Combination {
    /// True if this combination is a bomb (can be played on top of any non-bomb).
    #[inline]
    pub fn is_bomb(self) -> bool {
        matches!(self, Combination::Bomb)
    }

    /// True if this is a straight (any length).
    #[inline]
    pub fn is_straight(self) -> bool {
        use Combination::*;
        matches!(self, Straight5|Straight6|Straight7|Straight8|Straight9|Straight10|Straight11|Straight12|Straight13)
    }

    #[inline]
    pub fn is_double_straight(self) -> bool {
        use Combination::*;
        matches!(self, DoubleStraight2|DoubleStraight3|DoubleStraight4|DoubleStraight5|DoubleStraight6|DoubleStraight7|DoubleStraight8)
    }

    #[inline]
    pub fn is_triple_straight(self) -> bool {
        use Combination::*;
        matches!(self, TripleStraight2|TripleStraight3|TripleStraight4|TripleStraight5)
    }
}

/// An encoded game move.  rank=0, auxiliary=0 when combination==Pass.
/// rank values: 3-15 where 3-13=numeric, 14=Ace, 15=Two-as-high.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move {
    pub combination: Combination,
    /// Primary rank: 0 for pass, 3-15 otherwise.
    pub rank: u8,
    /// Auxiliary rank (pair in full house, kicker in bomb). 0 = none.
    pub auxiliary: u8,
}

impl Move {
    pub const PASS: Move = Move { combination: Combination::Pass, rank: 0, auxiliary: 0 };

    #[inline]
    pub fn new(combination: Combination, rank: u8, auxiliary: u8) -> Self {
        Move { combination, rank, auxiliary }
    }

    /// Decode a move ID (0..LEGAL_MOVES_SIZE) to a Move.
    pub fn decode(id: usize) -> Self {
        debug_assert!(id < LEGAL_MOVES_SIZE);
        if id == PASS {
            return Move::PASS;
        }
        if id < DOUBLE_START {
            let rank = (id - SINGLE_START + 3) as u8;
            return Move { combination: Combination::Single, rank, auxiliary: 0 };
        }
        if id < TRIPLE_START {
            let rank = (id - DOUBLE_START + 3) as u8;
            return Move { combination: Combination::Double, rank, auxiliary: 0 };
        }
        if id < FULL_HOUSE_START {
            let rank = (id - TRIPLE_START + 3) as u8;
            return Move { combination: Combination::Triple, rank, auxiliary: 0 };
        }
        if id < BOMB_START {
            let x = id - FULL_HOUSE_START;
            let triple_rank = (x / 11 + 3) as u8;
            let rem = x % 11;
            let aux = if rem < (triple_rank - 3) as usize { (rem + 3) as u8 } else { (rem + 4) as u8 };
            return Move { combination: Combination::FullHouse, rank: triple_rank, auxiliary: aux };
        }
        if id < STRAIGHT5_START {
            let x = id - BOMB_START;
            let bomb_rank = (x / 13 + 3) as u8;
            let rem = x % 13;
            let aux = if rem == 0 {
                0
            } else if rem + 2 < bomb_rank as usize {
                (rem + 2) as u8
            } else {
                (rem + 3) as u8
            };
            return Move { combination: Combination::Bomb, rank: bomb_rank, auxiliary: aux };
        }
        if id < STRAIGHT6_START {
            return Move { combination: Combination::Straight5, rank: (id - STRAIGHT5_START + 6) as u8, auxiliary: 0 };
        }
        if id < STRAIGHT7_START {
            return Move { combination: Combination::Straight6, rank: (id - STRAIGHT6_START + 7) as u8, auxiliary: 0 };
        }
        if id < STRAIGHT8_START {
            return Move { combination: Combination::Straight7, rank: (id - STRAIGHT7_START + 8) as u8, auxiliary: 0 };
        }
        if id < STRAIGHT9_START {
            return Move { combination: Combination::Straight8, rank: (id - STRAIGHT8_START + 9) as u8, auxiliary: 0 };
        }
        if id < STRAIGHT10_START {
            return Move { combination: Combination::Straight9, rank: (id - STRAIGHT9_START + 10) as u8, auxiliary: 0 };
        }
        if id < STRAIGHT11_START {
            return Move { combination: Combination::Straight10, rank: (id - STRAIGHT10_START + 11) as u8, auxiliary: 0 };
        }
        if id < STRAIGHT12_START {
            return Move { combination: Combination::Straight11, rank: (id - STRAIGHT11_START + 12) as u8, auxiliary: 0 };
        }
        if id < STRAIGHT13_START {
            return Move { combination: Combination::Straight12, rank: (id - STRAIGHT12_START + 13) as u8, auxiliary: 0 };
        }
        if id < DOUBLESTRAIGHT2_START {
            // Only one 13-card straight exists; rank 15 = top card is the 2.
            return Move { combination: Combination::Straight13, rank: 15, auxiliary: 0 };
        }
        if id < DOUBLESTRAIGHT3_START {
            return Move { combination: Combination::DoubleStraight2, rank: (id - DOUBLESTRAIGHT2_START + 4) as u8, auxiliary: 0 };
        }
        if id < DOUBLESTRAIGHT4_START {
            return Move { combination: Combination::DoubleStraight3, rank: (id - DOUBLESTRAIGHT3_START + 5) as u8, auxiliary: 0 };
        }
        if id < DOUBLESTRAIGHT5_START {
            return Move { combination: Combination::DoubleStraight4, rank: (id - DOUBLESTRAIGHT4_START + 6) as u8, auxiliary: 0 };
        }
        if id < DOUBLESTRAIGHT6_START {
            return Move { combination: Combination::DoubleStraight5, rank: (id - DOUBLESTRAIGHT5_START + 7) as u8, auxiliary: 0 };
        }
        if id < DOUBLESTRAIGHT7_START {
            return Move { combination: Combination::DoubleStraight6, rank: (id - DOUBLESTRAIGHT6_START + 8) as u8, auxiliary: 0 };
        }
        if id < TRIPLESTRAIGHT2_START {
            return Move { combination: Combination::DoubleStraight7, rank: (id - DOUBLESTRAIGHT7_START + 9) as u8, auxiliary: 0 };
        }
        if id < TRIPLESTRAIGHT3_START {
            return Move { combination: Combination::TripleStraight2, rank: (id - TRIPLESTRAIGHT2_START + 4) as u8, auxiliary: 0 };
        }
        if id < TRIPLESTRAIGHT4_START {
            return Move { combination: Combination::TripleStraight3, rank: (id - TRIPLESTRAIGHT3_START + 5) as u8, auxiliary: 0 };
        }
        if id < TRIPLESTRAIGHT5_START {
            return Move { combination: Combination::TripleStraight4, rank: (id - TRIPLESTRAIGHT4_START + 6) as u8, auxiliary: 0 };
        }
        if id < DOUBLESTRAIGHT8_START {
            return Move { combination: Combination::TripleStraight5, rank: (id - TRIPLESTRAIGHT5_START + 7) as u8, auxiliary: 0 };
        }
        Move { combination: Combination::DoubleStraight8, rank: (id - DOUBLESTRAIGHT8_START + 10) as u8, auxiliary: 0 }
    }

    /// Encode a Move back to its move ID.
    pub fn encode(self) -> usize {
        use Combination::*;
        match self.combination {
            Pass => PASS,
            Single => SINGLE_START + (self.rank - 3) as usize,
            Double => DOUBLE_START + (self.rank - 3) as usize,
            Triple => TRIPLE_START + (self.rank - 3) as usize,
            FullHouse => {
                let base = (self.rank - 3) as usize * 11;
                let offset = if self.auxiliary < self.rank {
                    (self.auxiliary - 3) as usize
                } else {
                    (self.auxiliary - 4) as usize
                };
                FULL_HOUSE_START + base + offset
            }
            Bomb => {
                if self.auxiliary == 0 {
                    return BOMB_START + (self.rank - 3) as usize * 13;
                }
                let base = (self.rank - 3) as usize * 13;
                let offset = if self.auxiliary < self.rank {
                    (self.auxiliary - 2) as usize
                } else {
                    (self.auxiliary - 3) as usize
                };
                BOMB_START + base + offset
            }
            Straight5  => STRAIGHT5_START  + (self.rank - 6) as usize,
            Straight6  => STRAIGHT6_START  + (self.rank - 7) as usize,
            Straight7  => STRAIGHT7_START  + (self.rank - 8) as usize,
            Straight8  => STRAIGHT8_START  + (self.rank - 9) as usize,
            Straight9  => STRAIGHT9_START  + (self.rank - 10) as usize,
            Straight10 => STRAIGHT10_START + (self.rank - 11) as usize,
            Straight11 => STRAIGHT11_START + (self.rank - 12) as usize,
            Straight12 => STRAIGHT12_START + (self.rank - 13) as usize,
            Straight13 => STRAIGHT13_START,
            DoubleStraight2 => DOUBLESTRAIGHT2_START + (self.rank - 4) as usize,
            DoubleStraight3 => DOUBLESTRAIGHT3_START + (self.rank - 5) as usize,
            DoubleStraight4 => DOUBLESTRAIGHT4_START + (self.rank - 6) as usize,
            DoubleStraight5 => DOUBLESTRAIGHT5_START + (self.rank - 7) as usize,
            DoubleStraight6 => DOUBLESTRAIGHT6_START + (self.rank - 8) as usize,
            DoubleStraight7 => DOUBLESTRAIGHT7_START + (self.rank - 9) as usize,
            DoubleStraight8 => DOUBLESTRAIGHT8_START + (self.rank - 10) as usize,
            TripleStraight2 => TRIPLESTRAIGHT2_START + (self.rank - 4) as usize,
            TripleStraight3 => TRIPLESTRAIGHT3_START + (self.rank - 5) as usize,
            TripleStraight4 => TRIPLESTRAIGHT4_START + (self.rank - 6) as usize,
            TripleStraight5 => TRIPLESTRAIGHT5_START + (self.rank - 7) as usize,
        }
    }

    /// Number of cards consumed by this move.
    pub fn num_cards(self) -> u8 {
        use Combination::*;
        match self.combination {
            Pass => 0,
            Single => 1,
            Double => 2,
            Triple => 3,
            FullHouse => 5,
            Bomb => (if self.rank == 14 { 3 } else { 4 }) + if self.auxiliary == 0 { 0 } else { 1 },
            Straight5 => 5,
            Straight6 => 6,
            Straight7 => 7,
            Straight8 => 8,
            Straight9 => 9,
            Straight10 => 10,
            Straight11 => 11,
            Straight12 => 12,
            Straight13 => 13,
            DoubleStraight2 => 4,
            DoubleStraight3 => 6,
            DoubleStraight4 => 8,
            DoubleStraight5 => 10,
            DoubleStraight6 => 12,
            DoubleStraight7 => 14,
            DoubleStraight8 => 16,
            TripleStraight2 => 6,
            TripleStraight3 => 9,
            TripleStraight4 => 12,
            TripleStraight5 => 15,
        }
    }
}

fn rank_char(rank: u8) -> char {
    match rank {
        2 | 15 => '2',
        3..=9  => (b'0' + rank) as char,
        10     => '0',
        11     => 'J',
        12     => 'Q',
        13     => 'K',
        _      => 'A',
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Combination::*;
        let r = rank_char(self.rank);
        let a = rank_char(self.auxiliary);
        match self.combination {
            Pass => write!(f, "PASS"),
            Single => write!(f, "{r}"),
            Double => write!(f, "{r}{r}"),
            Triple => write!(f, "{r}{r}{r}"),
            FullHouse => write!(f, "{r}{r}{r}{a}{a}"),
            Bomb => {
                if self.rank == 14 {
                    write!(f, "AAA")?;
                } else {
                    write!(f, "{r}{r}{r}{r}")?;
                }
                if self.auxiliary != 0 { write!(f, "{a}")?; }
                Ok(())
            }
            Straight5  => write_straight(f, self.rank, 5),
            Straight6  => write_straight(f, self.rank, 6),
            Straight7  => write_straight(f, self.rank, 7),
            Straight8  => write_straight(f, self.rank, 8),
            Straight9  => write_straight(f, self.rank, 9),
            Straight10 => write_straight(f, self.rank, 10),
            Straight11 => write_straight(f, self.rank, 11),
            Straight12 => write_straight(f, self.rank, 12),
            Straight13 => write!(f, "34567890JQKA2"),
            DoubleStraight2 => write_double_straight(f, self.rank, 2),
            DoubleStraight3 => write_double_straight(f, self.rank, 3),
            DoubleStraight4 => write_double_straight(f, self.rank, 4),
            DoubleStraight5 => write_double_straight(f, self.rank, 5),
            DoubleStraight6 => write_double_straight(f, self.rank, 6),
            DoubleStraight7 => write_double_straight(f, self.rank, 7),
            DoubleStraight8 => write_double_straight(f, self.rank, 8),
            TripleStraight2 => write_triple_straight(f, self.rank, 2),
            TripleStraight3 => write_triple_straight(f, self.rank, 3),
            TripleStraight4 => write_triple_straight(f, self.rank, 4),
            TripleStraight5 => write_triple_straight(f, self.rank, 5),
        }
    }
}

fn write_straight(f: &mut fmt::Formatter<'_>, top_rank: u8, len: u8) -> fmt::Result {
    for i in (0..len).rev() {
        write!(f, "{}", rank_char(top_rank - i))?;
    }
    Ok(())
}

fn write_double_straight(f: &mut fmt::Formatter<'_>, top_rank: u8, pairs: u8) -> fmt::Result {
    for i in (0..pairs).rev() {
        let r = rank_char(top_rank - i);
        write!(f, "{r}{r}")?;
    }
    Ok(())
}

fn write_triple_straight(f: &mut fmt::Formatter<'_>, top_rank: u8, triples: u8) -> fmt::Result {
    for i in (0..triples).rev() {
        let r = rank_char(top_rank - i);
        write!(f, "{r}{r}{r}")?;
    }
    Ok(())
}

pub const MOVE_TO_CARDS: [[u8; 14]; LEGAL_MOVES_SIZE] = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [2,0,0,0,0,0,0,0,0,0,0,0,0,2],
    [0,2,0,0,0,0,0,0,0,0,0,0,0,2],
    [0,0,2,0,0,0,0,0,0,0,0,0,0,2],
    [0,0,0,2,0,0,0,0,0,0,0,0,0,2],
    [0,0,0,0,2,0,0,0,0,0,0,0,0,2],
    [0,0,0,0,0,2,0,0,0,0,0,0,0,2],
    [0,0,0,0,0,0,2,0,0,0,0,0,0,2],
    [0,0,0,0,0,0,0,2,0,0,0,0,0,2],
    [0,0,0,0,0,0,0,0,2,0,0,0,0,2],
    [0,0,0,0,0,0,0,0,0,2,0,0,0,2],
    [0,0,0,0,0,0,0,0,0,0,2,0,0,2],
    [0,0,0,0,0,0,0,0,0,0,0,2,0,2],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [0,3,0,0,0,0,0,0,0,0,0,0,0,3],
    [0,0,3,0,0,0,0,0,0,0,0,0,0,3],
    [0,0,0,3,0,0,0,0,0,0,0,0,0,3],
    [0,0,0,0,3,0,0,0,0,0,0,0,0,3],
    [0,0,0,0,0,3,0,0,0,0,0,0,0,3],
    [0,0,0,0,0,0,3,0,0,0,0,0,0,3],
    [0,0,0,0,0,0,0,3,0,0,0,0,0,3],
    [0,0,0,0,0,0,0,0,3,0,0,0,0,3],
    [0,0,0,0,0,0,0,0,0,3,0,0,0,3],
    [0,0,0,0,0,0,0,0,0,0,3,0,0,3],
    [3,2,0,0,0,0,0,0,0,0,0,0,0,5],
    [3,0,2,0,0,0,0,0,0,0,0,0,0,5],
    [3,0,0,2,0,0,0,0,0,0,0,0,0,5],
    [3,0,0,0,2,0,0,0,0,0,0,0,0,5],
    [3,0,0,0,0,2,0,0,0,0,0,0,0,5],
    [3,0,0,0,0,0,2,0,0,0,0,0,0,5],
    [3,0,0,0,0,0,0,2,0,0,0,0,0,5],
    [3,0,0,0,0,0,0,0,2,0,0,0,0,5],
    [3,0,0,0,0,0,0,0,0,2,0,0,0,5],
    [3,0,0,0,0,0,0,0,0,0,2,0,0,5],
    [3,0,0,0,0,0,0,0,0,0,0,2,0,5],
    [2,3,0,0,0,0,0,0,0,0,0,0,0,5],
    [0,3,2,0,0,0,0,0,0,0,0,0,0,5],
    [0,3,0,2,0,0,0,0,0,0,0,0,0,5],
    [0,3,0,0,2,0,0,0,0,0,0,0,0,5],
    [0,3,0,0,0,2,0,0,0,0,0,0,0,5],
    [0,3,0,0,0,0,2,0,0,0,0,0,0,5],
    [0,3,0,0,0,0,0,2,0,0,0,0,0,5],
    [0,3,0,0,0,0,0,0,2,0,0,0,0,5],
    [0,3,0,0,0,0,0,0,0,2,0,0,0,5],
    [0,3,0,0,0,0,0,0,0,0,2,0,0,5],
    [0,3,0,0,0,0,0,0,0,0,0,2,0,5],
    [2,0,3,0,0,0,0,0,0,0,0,0,0,5],
    [0,2,3,0,0,0,0,0,0,0,0,0,0,5],
    [0,0,3,2,0,0,0,0,0,0,0,0,0,5],
    [0,0,3,0,2,0,0,0,0,0,0,0,0,5],
    [0,0,3,0,0,2,0,0,0,0,0,0,0,5],
    [0,0,3,0,0,0,2,0,0,0,0,0,0,5],
    [0,0,3,0,0,0,0,2,0,0,0,0,0,5],
    [0,0,3,0,0,0,0,0,2,0,0,0,0,5],
    [0,0,3,0,0,0,0,0,0,2,0,0,0,5],
    [0,0,3,0,0,0,0,0,0,0,2,0,0,5],
    [0,0,3,0,0,0,0,0,0,0,0,2,0,5],
    [2,0,0,3,0,0,0,0,0,0,0,0,0,5],
    [0,2,0,3,0,0,0,0,0,0,0,0,0,5],
    [0,0,2,3,0,0,0,0,0,0,0,0,0,5],
    [0,0,0,3,2,0,0,0,0,0,0,0,0,5],
    [0,0,0,3,0,2,0,0,0,0,0,0,0,5],
    [0,0,0,3,0,0,2,0,0,0,0,0,0,5],
    [0,0,0,3,0,0,0,2,0,0,0,0,0,5],
    [0,0,0,3,0,0,0,0,2,0,0,0,0,5],
    [0,0,0,3,0,0,0,0,0,2,0,0,0,5],
    [0,0,0,3,0,0,0,0,0,0,2,0,0,5],
    [0,0,0,3,0,0,0,0,0,0,0,2,0,5],
    [2,0,0,0,3,0,0,0,0,0,0,0,0,5],
    [0,2,0,0,3,0,0,0,0,0,0,0,0,5],
    [0,0,2,0,3,0,0,0,0,0,0,0,0,5],
    [0,0,0,2,3,0,0,0,0,0,0,0,0,5],
    [0,0,0,0,3,2,0,0,0,0,0,0,0,5],
    [0,0,0,0,3,0,2,0,0,0,0,0,0,5],
    [0,0,0,0,3,0,0,2,0,0,0,0,0,5],
    [0,0,0,0,3,0,0,0,2,0,0,0,0,5],
    [0,0,0,0,3,0,0,0,0,2,0,0,0,5],
    [0,0,0,0,3,0,0,0,0,0,2,0,0,5],
    [0,0,0,0,3,0,0,0,0,0,0,2,0,5],
    [2,0,0,0,0,3,0,0,0,0,0,0,0,5],
    [0,2,0,0,0,3,0,0,0,0,0,0,0,5],
    [0,0,2,0,0,3,0,0,0,0,0,0,0,5],
    [0,0,0,2,0,3,0,0,0,0,0,0,0,5],
    [0,0,0,0,2,3,0,0,0,0,0,0,0,5],
    [0,0,0,0,0,3,2,0,0,0,0,0,0,5],
    [0,0,0,0,0,3,0,2,0,0,0,0,0,5],
    [0,0,0,0,0,3,0,0,2,0,0,0,0,5],
    [0,0,0,0,0,3,0,0,0,2,0,0,0,5],
    [0,0,0,0,0,3,0,0,0,0,2,0,0,5],
    [0,0,0,0,0,3,0,0,0,0,0,2,0,5],
    [2,0,0,0,0,0,3,0,0,0,0,0,0,5],
    [0,2,0,0,0,0,3,0,0,0,0,0,0,5],
    [0,0,2,0,0,0,3,0,0,0,0,0,0,5],
    [0,0,0,2,0,0,3,0,0,0,0,0,0,5],
    [0,0,0,0,2,0,3,0,0,0,0,0,0,5],
    [0,0,0,0,0,2,3,0,0,0,0,0,0,5],
    [0,0,0,0,0,0,3,2,0,0,0,0,0,5],
    [0,0,0,0,0,0,3,0,2,0,0,0,0,5],
    [0,0,0,0,0,0,3,0,0,2,0,0,0,5],
    [0,0,0,0,0,0,3,0,0,0,2,0,0,5],
    [0,0,0,0,0,0,3,0,0,0,0,2,0,5],
    [2,0,0,0,0,0,0,3,0,0,0,0,0,5],
    [0,2,0,0,0,0,0,3,0,0,0,0,0,5],
    [0,0,2,0,0,0,0,3,0,0,0,0,0,5],
    [0,0,0,2,0,0,0,3,0,0,0,0,0,5],
    [0,0,0,0,2,0,0,3,0,0,0,0,0,5],
    [0,0,0,0,0,2,0,3,0,0,0,0,0,5],
    [0,0,0,0,0,0,2,3,0,0,0,0,0,5],
    [0,0,0,0,0,0,0,3,2,0,0,0,0,5],
    [0,0,0,0,0,0,0,3,0,2,0,0,0,5],
    [0,0,0,0,0,0,0,3,0,0,2,0,0,5],
    [0,0,0,0,0,0,0,3,0,0,0,2,0,5],
    [2,0,0,0,0,0,0,0,3,0,0,0,0,5],
    [0,2,0,0,0,0,0,0,3,0,0,0,0,5],
    [0,0,2,0,0,0,0,0,3,0,0,0,0,5],
    [0,0,0,2,0,0,0,0,3,0,0,0,0,5],
    [0,0,0,0,2,0,0,0,3,0,0,0,0,5],
    [0,0,0,0,0,2,0,0,3,0,0,0,0,5],
    [0,0,0,0,0,0,2,0,3,0,0,0,0,5],
    [0,0,0,0,0,0,0,2,3,0,0,0,0,5],
    [0,0,0,0,0,0,0,0,3,2,0,0,0,5],
    [0,0,0,0,0,0,0,0,3,0,2,0,0,5],
    [0,0,0,0,0,0,0,0,3,0,0,2,0,5],
    [2,0,0,0,0,0,0,0,0,3,0,0,0,5],
    [0,2,0,0,0,0,0,0,0,3,0,0,0,5],
    [0,0,2,0,0,0,0,0,0,3,0,0,0,5],
    [0,0,0,2,0,0,0,0,0,3,0,0,0,5],
    [0,0,0,0,2,0,0,0,0,3,0,0,0,5],
    [0,0,0,0,0,2,0,0,0,3,0,0,0,5],
    [0,0,0,0,0,0,2,0,0,3,0,0,0,5],
    [0,0,0,0,0,0,0,2,0,3,0,0,0,5],
    [0,0,0,0,0,0,0,0,2,3,0,0,0,5],
    [0,0,0,0,0,0,0,0,0,3,2,0,0,5],
    [0,0,0,0,0,0,0,0,0,3,0,2,0,5],
    [2,0,0,0,0,0,0,0,0,0,3,0,0,5],
    [0,2,0,0,0,0,0,0,0,0,3,0,0,5],
    [0,0,2,0,0,0,0,0,0,0,3,0,0,5],
    [0,0,0,2,0,0,0,0,0,0,3,0,0,5],
    [0,0,0,0,2,0,0,0,0,0,3,0,0,5],
    [0,0,0,0,0,2,0,0,0,0,3,0,0,5],
    [0,0,0,0,0,0,2,0,0,0,3,0,0,5],
    [0,0,0,0,0,0,0,2,0,0,3,0,0,5],
    [0,0,0,0,0,0,0,0,2,0,3,0,0,5],
    [0,0,0,0,0,0,0,0,0,2,3,0,0,5],
    [0,0,0,0,0,0,0,0,0,0,3,2,0,5],
    [2,0,0,0,0,0,0,0,0,0,0,3,0,5],
    [0,2,0,0,0,0,0,0,0,0,0,3,0,5],
    [0,0,2,0,0,0,0,0,0,0,0,3,0,5],
    [0,0,0,2,0,0,0,0,0,0,0,3,0,5],
    [0,0,0,0,2,0,0,0,0,0,0,3,0,5],
    [0,0,0,0,0,2,0,0,0,0,0,3,0,5],
    [0,0,0,0,0,0,2,0,0,0,0,3,0,5],
    [0,0,0,0,0,0,0,2,0,0,0,3,0,5],
    [0,0,0,0,0,0,0,0,2,0,0,3,0,5],
    [0,0,0,0,0,0,0,0,0,2,0,3,0,5],
    [0,0,0,0,0,0,0,0,0,0,2,3,0,5],
    [4,0,0,0,0,0,0,0,0,0,0,0,0,4],
    [4,1,0,0,0,0,0,0,0,0,0,0,0,5],
    [4,0,1,0,0,0,0,0,0,0,0,0,0,5],
    [4,0,0,1,0,0,0,0,0,0,0,0,0,5],
    [4,0,0,0,1,0,0,0,0,0,0,0,0,5],
    [4,0,0,0,0,1,0,0,0,0,0,0,0,5],
    [4,0,0,0,0,0,1,0,0,0,0,0,0,5],
    [4,0,0,0,0,0,0,1,0,0,0,0,0,5],
    [4,0,0,0,0,0,0,0,1,0,0,0,0,5],
    [4,0,0,0,0,0,0,0,0,1,0,0,0,5],
    [4,0,0,0,0,0,0,0,0,0,1,0,0,5],
    [4,0,0,0,0,0,0,0,0,0,0,1,0,5],
    [4,0,0,0,0,0,0,0,0,0,0,0,1,5],
    [0,4,0,0,0,0,0,0,0,0,0,0,0,4],
    [1,4,0,0,0,0,0,0,0,0,0,0,0,5],
    [0,4,1,0,0,0,0,0,0,0,0,0,0,5],
    [0,4,0,1,0,0,0,0,0,0,0,0,0,5],
    [0,4,0,0,1,0,0,0,0,0,0,0,0,5],
    [0,4,0,0,0,1,0,0,0,0,0,0,0,5],
    [0,4,0,0,0,0,1,0,0,0,0,0,0,5],
    [0,4,0,0,0,0,0,1,0,0,0,0,0,5],
    [0,4,0,0,0,0,0,0,1,0,0,0,0,5],
    [0,4,0,0,0,0,0,0,0,1,0,0,0,5],
    [0,4,0,0,0,0,0,0,0,0,1,0,0,5],
    [0,4,0,0,0,0,0,0,0,0,0,1,0,5],
    [0,4,0,0,0,0,0,0,0,0,0,0,1,5],
    [0,0,4,0,0,0,0,0,0,0,0,0,0,4],
    [1,0,4,0,0,0,0,0,0,0,0,0,0,5],
    [0,1,4,0,0,0,0,0,0,0,0,0,0,5],
    [0,0,4,1,0,0,0,0,0,0,0,0,0,5],
    [0,0,4,0,1,0,0,0,0,0,0,0,0,5],
    [0,0,4,0,0,1,0,0,0,0,0,0,0,5],
    [0,0,4,0,0,0,1,0,0,0,0,0,0,5],
    [0,0,4,0,0,0,0,1,0,0,0,0,0,5],
    [0,0,4,0,0,0,0,0,1,0,0,0,0,5],
    [0,0,4,0,0,0,0,0,0,1,0,0,0,5],
    [0,0,4,0,0,0,0,0,0,0,1,0,0,5],
    [0,0,4,0,0,0,0,0,0,0,0,1,0,5],
    [0,0,4,0,0,0,0,0,0,0,0,0,1,5],
    [0,0,0,4,0,0,0,0,0,0,0,0,0,4],
    [1,0,0,4,0,0,0,0,0,0,0,0,0,5],
    [0,1,0,4,0,0,0,0,0,0,0,0,0,5],
    [0,0,1,4,0,0,0,0,0,0,0,0,0,5],
    [0,0,0,4,1,0,0,0,0,0,0,0,0,5],
    [0,0,0,4,0,1,0,0,0,0,0,0,0,5],
    [0,0,0,4,0,0,1,0,0,0,0,0,0,5],
    [0,0,0,4,0,0,0,1,0,0,0,0,0,5],
    [0,0,0,4,0,0,0,0,1,0,0,0,0,5],
    [0,0,0,4,0,0,0,0,0,1,0,0,0,5],
    [0,0,0,4,0,0,0,0,0,0,1,0,0,5],
    [0,0,0,4,0,0,0,0,0,0,0,1,0,5],
    [0,0,0,4,0,0,0,0,0,0,0,0,1,5],
    [0,0,0,0,4,0,0,0,0,0,0,0,0,4],
    [1,0,0,0,4,0,0,0,0,0,0,0,0,5],
    [0,1,0,0,4,0,0,0,0,0,0,0,0,5],
    [0,0,1,0,4,0,0,0,0,0,0,0,0,5],
    [0,0,0,1,4,0,0,0,0,0,0,0,0,5],
    [0,0,0,0,4,1,0,0,0,0,0,0,0,5],
    [0,0,0,0,4,0,1,0,0,0,0,0,0,5],
    [0,0,0,0,4,0,0,1,0,0,0,0,0,5],
    [0,0,0,0,4,0,0,0,1,0,0,0,0,5],
    [0,0,0,0,4,0,0,0,0,1,0,0,0,5],
    [0,0,0,0,4,0,0,0,0,0,1,0,0,5],
    [0,0,0,0,4,0,0,0,0,0,0,1,0,5],
    [0,0,0,0,4,0,0,0,0,0,0,0,1,5],
    [0,0,0,0,0,4,0,0,0,0,0,0,0,4],
    [1,0,0,0,0,4,0,0,0,0,0,0,0,5],
    [0,1,0,0,0,4,0,0,0,0,0,0,0,5],
    [0,0,1,0,0,4,0,0,0,0,0,0,0,5],
    [0,0,0,1,0,4,0,0,0,0,0,0,0,5],
    [0,0,0,0,1,4,0,0,0,0,0,0,0,5],
    [0,0,0,0,0,4,1,0,0,0,0,0,0,5],
    [0,0,0,0,0,4,0,1,0,0,0,0,0,5],
    [0,0,0,0,0,4,0,0,1,0,0,0,0,5],
    [0,0,0,0,0,4,0,0,0,1,0,0,0,5],
    [0,0,0,0,0,4,0,0,0,0,1,0,0,5],
    [0,0,0,0,0,4,0,0,0,0,0,1,0,5],
    [0,0,0,0,0,4,0,0,0,0,0,0,1,5],
    [0,0,0,0,0,0,4,0,0,0,0,0,0,4],
    [1,0,0,0,0,0,4,0,0,0,0,0,0,5],
    [0,1,0,0,0,0,4,0,0,0,0,0,0,5],
    [0,0,1,0,0,0,4,0,0,0,0,0,0,5],
    [0,0,0,1,0,0,4,0,0,0,0,0,0,5],
    [0,0,0,0,1,0,4,0,0,0,0,0,0,5],
    [0,0,0,0,0,1,4,0,0,0,0,0,0,5],
    [0,0,0,0,0,0,4,1,0,0,0,0,0,5],
    [0,0,0,0,0,0,4,0,1,0,0,0,0,5],
    [0,0,0,0,0,0,4,0,0,1,0,0,0,5],
    [0,0,0,0,0,0,4,0,0,0,1,0,0,5],
    [0,0,0,0,0,0,4,0,0,0,0,1,0,5],
    [0,0,0,0,0,0,4,0,0,0,0,0,1,5],
    [0,0,0,0,0,0,0,4,0,0,0,0,0,4],
    [1,0,0,0,0,0,0,4,0,0,0,0,0,5],
    [0,1,0,0,0,0,0,4,0,0,0,0,0,5],
    [0,0,1,0,0,0,0,4,0,0,0,0,0,5],
    [0,0,0,1,0,0,0,4,0,0,0,0,0,5],
    [0,0,0,0,1,0,0,4,0,0,0,0,0,5],
    [0,0,0,0,0,1,0,4,0,0,0,0,0,5],
    [0,0,0,0,0,0,1,4,0,0,0,0,0,5],
    [0,0,0,0,0,0,0,4,1,0,0,0,0,5],
    [0,0,0,0,0,0,0,4,0,1,0,0,0,5],
    [0,0,0,0,0,0,0,4,0,0,1,0,0,5],
    [0,0,0,0,0,0,0,4,0,0,0,1,0,5],
    [0,0,0,0,0,0,0,4,0,0,0,0,1,5],
    [0,0,0,0,0,0,0,0,4,0,0,0,0,4],
    [1,0,0,0,0,0,0,0,4,0,0,0,0,5],
    [0,1,0,0,0,0,0,0,4,0,0,0,0,5],
    [0,0,1,0,0,0,0,0,4,0,0,0,0,5],
    [0,0,0,1,0,0,0,0,4,0,0,0,0,5],
    [0,0,0,0,1,0,0,0,4,0,0,0,0,5],
    [0,0,0,0,0,1,0,0,4,0,0,0,0,5],
    [0,0,0,0,0,0,1,0,4,0,0,0,0,5],
    [0,0,0,0,0,0,0,1,4,0,0,0,0,5],
    [0,0,0,0,0,0,0,0,4,1,0,0,0,5],
    [0,0,0,0,0,0,0,0,4,0,1,0,0,5],
    [0,0,0,0,0,0,0,0,4,0,0,1,0,5],
    [0,0,0,0,0,0,0,0,4,0,0,0,1,5],
    [0,0,0,0,0,0,0,0,0,4,0,0,0,4],
    [1,0,0,0,0,0,0,0,0,4,0,0,0,5],
    [0,1,0,0,0,0,0,0,0,4,0,0,0,5],
    [0,0,1,0,0,0,0,0,0,4,0,0,0,5],
    [0,0,0,1,0,0,0,0,0,4,0,0,0,5],
    [0,0,0,0,1,0,0,0,0,4,0,0,0,5],
    [0,0,0,0,0,1,0,0,0,4,0,0,0,5],
    [0,0,0,0,0,0,1,0,0,4,0,0,0,5],
    [0,0,0,0,0,0,0,1,0,4,0,0,0,5],
    [0,0,0,0,0,0,0,0,1,4,0,0,0,5],
    [0,0,0,0,0,0,0,0,0,4,1,0,0,5],
    [0,0,0,0,0,0,0,0,0,4,0,1,0,5],
    [0,0,0,0,0,0,0,0,0,4,0,0,1,5],
    [0,0,0,0,0,0,0,0,0,0,4,0,0,4],
    [1,0,0,0,0,0,0,0,0,0,4,0,0,5],
    [0,1,0,0,0,0,0,0,0,0,4,0,0,5],
    [0,0,1,0,0,0,0,0,0,0,4,0,0,5],
    [0,0,0,1,0,0,0,0,0,0,4,0,0,5],
    [0,0,0,0,1,0,0,0,0,0,4,0,0,5],
    [0,0,0,0,0,1,0,0,0,0,4,0,0,5],
    [0,0,0,0,0,0,1,0,0,0,4,0,0,5],
    [0,0,0,0,0,0,0,1,0,0,4,0,0,5],
    [0,0,0,0,0,0,0,0,1,0,4,0,0,5],
    [0,0,0,0,0,0,0,0,0,1,4,0,0,5],
    [0,0,0,0,0,0,0,0,0,0,4,1,0,5],
    [0,0,0,0,0,0,0,0,0,0,4,0,1,5],
    [0,0,0,0,0,0,0,0,0,0,0,3,0,3],
    [1,0,0,0,0,0,0,0,0,0,0,3,0,4],
    [0,1,0,0,0,0,0,0,0,0,0,3,0,4],
    [0,0,1,0,0,0,0,0,0,0,0,3,0,4],
    [0,0,0,1,0,0,0,0,0,0,0,3,0,4],
    [0,0,0,0,1,0,0,0,0,0,0,3,0,4],
    [0,0,0,0,0,1,0,0,0,0,0,3,0,4],
    [0,0,0,0,0,0,1,0,0,0,0,3,0,4],
    [0,0,0,0,0,0,0,1,0,0,0,3,0,4],
    [0,0,0,0,0,0,0,0,1,0,0,3,0,4],
    [0,0,0,0,0,0,0,0,0,1,0,3,0,4],
    [0,0,0,0,0,0,0,0,0,0,1,3,0,4],
    [0,0,0,0,0,0,0,0,0,0,0,3,1,4],
    [1,1,1,1,0,0,0,0,0,0,0,0,1,5],
    [1,1,1,1,1,0,0,0,0,0,0,0,0,5],
    [0,1,1,1,1,1,0,0,0,0,0,0,0,5],
    [0,0,1,1,1,1,1,0,0,0,0,0,0,5],
    [0,0,0,1,1,1,1,1,0,0,0,0,0,5],
    [0,0,0,0,1,1,1,1,1,0,0,0,0,5],
    [0,0,0,0,0,1,1,1,1,1,0,0,0,5],
    [0,0,0,0,0,0,1,1,1,1,1,0,0,5],
    [0,0,0,0,0,0,0,1,1,1,1,1,0,5],
    [0,0,0,0,0,0,0,0,1,1,1,1,1,5],
    [1,1,1,1,1,0,0,0,0,0,0,0,1,6],
    [1,1,1,1,1,1,0,0,0,0,0,0,0,6],
    [0,1,1,1,1,1,1,0,0,0,0,0,0,6],
    [0,0,1,1,1,1,1,1,0,0,0,0,0,6],
    [0,0,0,1,1,1,1,1,1,0,0,0,0,6],
    [0,0,0,0,1,1,1,1,1,1,0,0,0,6],
    [0,0,0,0,0,1,1,1,1,1,1,0,0,6],
    [0,0,0,0,0,0,1,1,1,1,1,1,0,6],
    [0,0,0,0,0,0,0,1,1,1,1,1,1,6],
    [1,1,1,1,1,1,0,0,0,0,0,0,1,7],
    [1,1,1,1,1,1,1,0,0,0,0,0,0,7],
    [0,1,1,1,1,1,1,1,0,0,0,0,0,7],
    [0,0,1,1,1,1,1,1,1,0,0,0,0,7],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,7],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,7],
    [0,0,0,0,0,1,1,1,1,1,1,1,0,7],
    [0,0,0,0,0,0,1,1,1,1,1,1,1,7],
    [1,1,1,1,1,1,1,0,0,0,0,0,1,8],
    [1,1,1,1,1,1,1,1,0,0,0,0,0,8],
    [0,1,1,1,1,1,1,1,1,0,0,0,0,8],
    [0,0,1,1,1,1,1,1,1,1,0,0,0,8],
    [0,0,0,1,1,1,1,1,1,1,1,0,0,8],
    [0,0,0,0,1,1,1,1,1,1,1,1,0,8],
    [0,0,0,0,0,1,1,1,1,1,1,1,1,8],
    [1,1,1,1,1,1,1,1,0,0,0,0,1,9],
    [1,1,1,1,1,1,1,1,1,0,0,0,0,9],
    [0,1,1,1,1,1,1,1,1,1,0,0,0,9],
    [0,0,1,1,1,1,1,1,1,1,1,0,0,9],
    [0,0,0,1,1,1,1,1,1,1,1,1,0,9],
    [0,0,0,0,1,1,1,1,1,1,1,1,1,9],
    [1,1,1,1,1,1,1,1,1,0,0,0,1,10],
    [1,1,1,1,1,1,1,1,1,1,0,0,0,10],
    [0,1,1,1,1,1,1,1,1,1,1,0,0,10],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,10],
    [0,0,0,1,1,1,1,1,1,1,1,1,1,10],
    [1,1,1,1,1,1,1,1,1,1,0,0,1,11],
    [1,1,1,1,1,1,1,1,1,1,1,0,0,11],
    [0,1,1,1,1,1,1,1,1,1,1,1,0,11],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,11],
    [1,1,1,1,1,1,1,1,1,1,1,0,1,12],
    [1,1,1,1,1,1,1,1,1,1,1,1,0,12],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,12],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,13],
    [2,2,0,0,0,0,0,0,0,0,0,0,0,4],
    [0,2,2,0,0,0,0,0,0,0,0,0,0,4],
    [0,0,2,2,0,0,0,0,0,0,0,0,0,4],
    [0,0,0,2,2,0,0,0,0,0,0,0,0,4],
    [0,0,0,0,2,2,0,0,0,0,0,0,0,4],
    [0,0,0,0,0,2,2,0,0,0,0,0,0,4],
    [0,0,0,0,0,0,2,2,0,0,0,0,0,4],
    [0,0,0,0,0,0,0,2,2,0,0,0,0,4],
    [0,0,0,0,0,0,0,0,2,2,0,0,0,4],
    [0,0,0,0,0,0,0,0,0,2,2,0,0,4],
    [0,0,0,0,0,0,0,0,0,0,2,2,0,4],
    [2,2,2,0,0,0,0,0,0,0,0,0,0,6],
    [0,2,2,2,0,0,0,0,0,0,0,0,0,6],
    [0,0,2,2,2,0,0,0,0,0,0,0,0,6],
    [0,0,0,2,2,2,0,0,0,0,0,0,0,6],
    [0,0,0,0,2,2,2,0,0,0,0,0,0,6],
    [0,0,0,0,0,2,2,2,0,0,0,0,0,6],
    [0,0,0,0,0,0,2,2,2,0,0,0,0,6],
    [0,0,0,0,0,0,0,2,2,2,0,0,0,6],
    [0,0,0,0,0,0,0,0,2,2,2,0,0,6],
    [0,0,0,0,0,0,0,0,0,2,2,2,0,6],
    [2,2,2,2,0,0,0,0,0,0,0,0,0,8],
    [0,2,2,2,2,0,0,0,0,0,0,0,0,8],
    [0,0,2,2,2,2,0,0,0,0,0,0,0,8],
    [0,0,0,2,2,2,2,0,0,0,0,0,0,8],
    [0,0,0,0,2,2,2,2,0,0,0,0,0,8],
    [0,0,0,0,0,2,2,2,2,0,0,0,0,8],
    [0,0,0,0,0,0,2,2,2,2,0,0,0,8],
    [0,0,0,0,0,0,0,2,2,2,2,0,0,8],
    [0,0,0,0,0,0,0,0,2,2,2,2,0,8],
    [2,2,2,2,2,0,0,0,0,0,0,0,0,10],
    [0,2,2,2,2,2,0,0,0,0,0,0,0,10],
    [0,0,2,2,2,2,2,0,0,0,0,0,0,10],
    [0,0,0,2,2,2,2,2,0,0,0,0,0,10],
    [0,0,0,0,2,2,2,2,2,0,0,0,0,10],
    [0,0,0,0,0,2,2,2,2,2,0,0,0,10],
    [0,0,0,0,0,0,2,2,2,2,2,0,0,10],
    [0,0,0,0,0,0,0,2,2,2,2,2,0,10],
    [2,2,2,2,2,2,0,0,0,0,0,0,0,12],
    [0,2,2,2,2,2,2,0,0,0,0,0,0,12],
    [0,0,2,2,2,2,2,2,0,0,0,0,0,12],
    [0,0,0,2,2,2,2,2,2,0,0,0,0,12],
    [0,0,0,0,2,2,2,2,2,2,0,0,0,12],
    [0,0,0,0,0,2,2,2,2,2,2,0,0,12],
    [0,0,0,0,0,0,2,2,2,2,2,2,0,12],
    [2,2,2,2,2,2,2,0,0,0,0,0,0,14],
    [0,2,2,2,2,2,2,2,0,0,0,0,0,14],
    [0,0,2,2,2,2,2,2,2,0,0,0,0,14],
    [0,0,0,2,2,2,2,2,2,2,0,0,0,14],
    [0,0,0,0,2,2,2,2,2,2,2,0,0,14],
    [0,0,0,0,0,2,2,2,2,2,2,2,0,14],
    [3,3,0,0,0,0,0,0,0,0,0,0,0,6],
    [0,3,3,0,0,0,0,0,0,0,0,0,0,6],
    [0,0,3,3,0,0,0,0,0,0,0,0,0,6],
    [0,0,0,3,3,0,0,0,0,0,0,0,0,6],
    [0,0,0,0,3,3,0,0,0,0,0,0,0,6],
    [0,0,0,0,0,3,3,0,0,0,0,0,0,6],
    [0,0,0,0,0,0,3,3,0,0,0,0,0,6],
    [0,0,0,0,0,0,0,3,3,0,0,0,0,6],
    [0,0,0,0,0,0,0,0,3,3,0,0,0,6],
    [0,0,0,0,0,0,0,0,0,3,3,0,0,6],
    [3,3,3,0,0,0,0,0,0,0,0,0,0,9],
    [0,3,3,3,0,0,0,0,0,0,0,0,0,9],
    [0,0,3,3,3,0,0,0,0,0,0,0,0,9],
    [0,0,0,3,3,3,0,0,0,0,0,0,0,9],
    [0,0,0,0,3,3,3,0,0,0,0,0,0,9],
    [0,0,0,0,0,3,3,3,0,0,0,0,0,9],
    [0,0,0,0,0,0,3,3,3,0,0,0,0,9],
    [0,0,0,0,0,0,0,3,3,3,0,0,0,9],
    [0,0,0,0,0,0,0,0,3,3,3,0,0,9],
    [3,3,3,3,0,0,0,0,0,0,0,0,0,12],
    [0,3,3,3,3,0,0,0,0,0,0,0,0,12],
    [0,0,3,3,3,3,0,0,0,0,0,0,0,12],
    [0,0,0,3,3,3,3,0,0,0,0,0,0,12],
    [0,0,0,0,3,3,3,3,0,0,0,0,0,12],
    [0,0,0,0,0,3,3,3,3,0,0,0,0,12],
    [0,0,0,0,0,0,3,3,3,3,0,0,0,12],
    [0,0,0,0,0,0,0,3,3,3,3,0,0,12],
    [3,3,3,3,3,0,0,0,0,0,0,0,0,15],
    [0,3,3,3,3,3,0,0,0,0,0,0,0,15],
    [0,0,3,3,3,3,3,0,0,0,0,0,0,15],
    [0,0,0,3,3,3,3,3,0,0,0,0,0,15],
    [0,0,0,0,3,3,3,3,3,0,0,0,0,15],
    [0,0,0,0,0,3,3,3,3,3,0,0,0,15],
    [0,0,0,0,0,0,3,3,3,3,3,0,0,15],
    [2,2,2,2,2,2,2,2,0,0,0,0,0,16],
    [0,2,2,2,2,2,2,2,2,0,0,0,0,16],
    [0,0,2,2,2,2,2,2,2,2,0,0,0,16],
    [0,0,0,2,2,2,2,2,2,2,2,0,0,16],
    [0,0,0,0,2,2,2,2,2,2,2,2,0,16],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        for id in 0..LEGAL_MOVES_SIZE {
            let m = Move::decode(id);
            let re = m.encode();
            assert_eq!(re, id, "roundtrip failed at id {id}");
        }
    }

    #[test]
    fn test_num_cards_matches_table() {
        for id in 0..LEGAL_MOVES_SIZE {
            let m = Move::decode(id);
            let table_total = MOVE_TO_CARDS[id][13];
            assert_eq!(m.num_cards(), table_total,
                "num_cards mismatch at id {id}: move says {}, table says {table_total}", m.num_cards());
        }
    }

    #[test]
    fn test_move_to_cards_limits() {
        for id in 0..LEGAL_MOVES_SIZE {
            let row = MOVE_TO_CARDS[id];
            let mut sum = 0u8;
            for rank in 0..13 {
                let count = row[rank];
                let limit = crate::consts::max_copies_in_deck(rank);
                assert!(count <= limit, "id {id} rank {rank}: count {count} > limit {limit}");
                sum += count;
            }
            assert_eq!(sum, row[13], "id {id}: sum {sum} != total {}", row[13]);
        }
    }

    #[test]
    fn test_pass() {
        let p = Move::PASS;
        assert_eq!(p.num_cards(), 0);
        assert_eq!(p.encode(), PASS);
        let back = Move::decode(PASS);
        assert_eq!(back.combination, Combination::Pass);
    }

    #[test]
    fn test_singles() {
        for rank in 3u8..=15 {
            let s = Move::new(Combination::Single, rank, 0);
            let id = s.encode();
            let back = Move::decode(id);
            assert_eq!(back.combination, Combination::Single);
            assert_eq!(back.rank, rank);
            assert_eq!(back.num_cards(), 1);
        }
    }

    #[test]
    fn test_full_house_sample() {
        let fh = Move::new(Combination::FullHouse, 5, 7);
        let id = fh.encode();
        assert!(id >= FULL_HOUSE_START && id < BOMB_START);
        let back = Move::decode(id);
        assert_eq!(back.combination, Combination::FullHouse);
        assert_eq!(back.rank, 5);
        assert_eq!(back.auxiliary, 7);
        assert_eq!(back.num_cards(), 5);

        let fh2 = Move::new(Combination::FullHouse, 7, 5);
        let id2 = fh2.encode();
        assert_ne!(id2, id);
        let back2 = Move::decode(id2);
        assert_eq!(back2.rank, 7);
        assert_eq!(back2.auxiliary, 5);
    }

    #[test]
    fn test_all_ids_in_range() {
        for id in 0..LEGAL_MOVES_SIZE {
            let m = Move::decode(id);
            let re = m.encode();
            assert!(re < LEGAL_MOVES_SIZE);
        }
    }
}
