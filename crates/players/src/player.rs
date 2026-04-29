use big2_core::{Game, Move, PartialGame, peek_tablebase_move};

/// Inner strategy interface.  Implementations must NOT call `state.apply_move`.
pub trait Strategy: Send {
    fn reset(&mut self) {}
    fn on_deal(&mut self, _player_num: usize) {}
    fn on_opponent_move(&mut self, _mv: Move) {}
    fn select_move_impl(&mut self, state: &PartialGame) -> Move;
}

/// Wraps a `Strategy` with the shared PartialGame state-tracking and
/// tablebase check that all players go through.
pub struct Player<S: Strategy> {
    pub(crate) state: PartialGame,
    pub(crate) strategy: S,
    player_num: usize,
    pub last_tb_case: i8,
    pub last_tb_forced_seq_len: usize,
    pub last_tb_opp1_table_straight: bool,
}

impl<S: Strategy> Player<S> {
    pub fn new(strategy: S) -> Self {
        Player {
            state: PartialGame::from_game(&Game::new(), 0),
            strategy,
            player_num: 0,
            last_tb_case: -1,
            last_tb_forced_seq_len: 0,
            last_tb_opp1_table_straight: false,
        }
    }

    pub fn reset(&mut self) {
        self.strategy.reset();
    }

    pub fn accept_deal(&mut self, game: &Game, player_num: usize) {
        self.state = PartialGame::from_game(game, player_num);
        self.player_num = player_num;
        self.strategy.on_deal(player_num);
    }

    pub fn accept_opponent_move(&mut self, mv: Move) {
        self.state.apply_move(mv);
        self.strategy.on_opponent_move(mv);
    }

    pub fn select_move(&mut self) -> Move {
        self.last_tb_case = -1;
        self.last_tb_forced_seq_len = 0;
        self.last_tb_opp1_table_straight = false;

        let tb = peek_tablebase_move(&self.state);
        self.last_tb_case = tb.tb_case;
        self.last_tb_forced_seq_len = tb.tb_forced_seq_len;
        self.last_tb_opp1_table_straight = tb.tb_opp1_table_straight;

        let mv = if let Some(m) = tb.mv {
            m
        } else {
            let state = &self.state;
            self.strategy.select_move_impl(state)
        };
        self.state.apply_move(mv);
        mv
    }
}

/// Type-erased player handle — returned by the registry, used by the simulator.
pub trait AnyPlayer: Send {
    fn reset(&mut self);
    fn accept_deal(&mut self, game: &Game, player_num: usize);
    fn accept_opponent_move(&mut self, mv: Move);
    fn select_move(&mut self) -> Move;
    fn last_tb_case(&self) -> i8;
    fn last_tb_forced_seq_len(&self) -> usize;
    fn last_tb_opp1_table_straight(&self) -> bool;
}

impl<S: Strategy> AnyPlayer for Player<S> {
    fn reset(&mut self) { Player::reset(self); }
    fn accept_deal(&mut self, game: &Game, player_num: usize) { Player::accept_deal(self, game, player_num); }
    fn accept_opponent_move(&mut self, mv: Move) { Player::accept_opponent_move(self, mv); }
    fn select_move(&mut self) -> Move { Player::select_move(self) }
    fn last_tb_case(&self) -> i8 { self.last_tb_case }
    fn last_tb_forced_seq_len(&self) -> usize { self.last_tb_forced_seq_len }
    fn last_tb_opp1_table_straight(&self) -> bool { self.last_tb_opp1_table_straight }
}
