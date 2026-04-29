use crate::game::Game;
use crate::moves::Move;
use crate::partial_game::PartialGame;

/// Everything recorded about a single turn for training / analysis.
pub struct TurnRecord {
    pub current_player: usize,
    pub game: Game,
    pub views: [PartialGame; 2],
    pub legal_moves: Vec<usize>,
    pub possible_moves: Vec<usize>,
    pub mv: Move,
    /// -1 = no tablebase, 1 = forced-win sequence, 2 = opponent has 1 card.
    pub tb_case: i8,
    pub tb_forced_seq_len: usize,
    pub tb_opp1_table_straight: bool,
}

#[derive(Default)]
pub struct TablebaseFirstHitStats {
    pub case1: u32,
    pub case2: u32,
    pub case1_seq_sum: u64,
    pub case1_seq_max: u32,
    pub case2_table_straight: u32,
}

pub struct GameRecord {
    initial_game: Game,
    views: [PartialGame; 2],
    pub turns: Vec<TurnRecord>,
}

impl GameRecord {
    pub fn new(game: &Game) -> Self {
        GameRecord {
            initial_game: game.clone(),
            views: [
                PartialGame::from_game(game, 0),
                PartialGame::from_game(game, 1),
            ],
            turns: Vec::new(),
        }
    }

    pub fn add_move(
        &mut self,
        mv: Move,
        game: &Game,
        tb_case: i8,
        tb_forced_seq_len: usize,
        tb_opp1_table_straight: bool,
    ) {
        let cp = game.current_player();
        let legal = self.views[cp].legal_moves();
        let possible = self.views[1 - cp].possible_moves();

        self.turns.push(TurnRecord {
            current_player: cp,
            game: game.clone(),
            views: self.views.clone(),
            legal_moves: legal,
            possible_moves: possible,
            mv,
            tb_case,
            tb_forced_seq_len,
            tb_opp1_table_straight,
        });

        self.views[0].apply_move(mv);
        self.views[1].apply_move(mv);
    }

    pub fn initial_game(&self) -> &Game {
        &self.initial_game
    }
}

pub fn tablebase_first_hit_stats(record: &GameRecord) -> TablebaseFirstHitStats {
    let mut stats = TablebaseFirstHitStats::default();
    let mut last_tb = [-1i8; 2];

    for turn in &record.turns {
        let p = turn.current_player;
        if turn.tb_case == 1 {
            if last_tb[p] != 1 {
                stats.case1 += 1;
                stats.case1_seq_sum += turn.tb_forced_seq_len as u64;
                if turn.tb_forced_seq_len as u32 > stats.case1_seq_max {
                    stats.case1_seq_max = turn.tb_forced_seq_len as u32;
                }
            }
            last_tb[p] = 1;
        } else if turn.tb_case == 2 {
            if last_tb[p] != 2 {
                stats.case2 += 1;
                if turn.tb_opp1_table_straight {
                    stats.case2_table_straight += 1;
                }
            }
            last_tb[p] = 2;
        } else {
            last_tb[p] = -1;
        }
    }
    stats
}
