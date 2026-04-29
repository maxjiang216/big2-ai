pub mod simulator;
pub mod coordinator;

pub use simulator::{play_game, play_game_shuffle, eval_match};
pub use coordinator::{run_games_parallel, run_paired_deals, PlayerFactory};
