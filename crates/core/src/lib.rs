pub mod consts;
pub mod moves;
pub mod game;
pub mod partial_game;
pub mod util;
pub mod tablebase;
pub mod game_record;

// Re-export the most commonly used items at the crate root.
pub use consts::{LEGAL_MOVES_SIZE, PASS, max_copies_in_deck, rank_to_idx};
pub use moves::{Combination, Move, MOVE_TO_CARDS};
pub use game::Game;
pub use partial_game::PartialGame;
pub use game_record::{GameRecord, TurnRecord, TablebaseFirstHitStats, tablebase_first_hit_stats};
pub use tablebase::{load_tablebase_opp1, peek_tablebase_move, TablebasePeekResult};
pub use util::{compute_legal_moves, compute_possible_moves, find_forced_win, opponent_can_respond};
