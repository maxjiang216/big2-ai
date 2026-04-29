use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use big2_core::GameRecord;
use big2_players::player::AnyPlayer;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::simulator::play_game_shuffle;

/// Factory function that constructs a fresh player for a given seed.
pub type PlayerFactory = Box<dyn Fn(u64) -> Box<dyn AnyPlayer> + Send + Sync>;

/// Run `n_games` self-play games in parallel using rayon.
///
/// Each game gets a deterministic seed = `base_seed + game_index` for
/// reproducibility.  Returns all records with their game indices.
pub fn run_games_parallel(
    factory0: &PlayerFactory,
    factory1: &PlayerFactory,
    n_games: u32,
    base_seed: u64,
    progress: Option<&AtomicU32>,
) -> Vec<(u32, GameRecord)> {
    (0..n_games)
        .into_par_iter()
        .map(|i| {
            let seed = base_seed.wrapping_add(i as u64);
            let mut p0 = factory0(seed);
            let mut p1 = factory1(seed ^ 0xDEAD_BEEF);
            let mut rng = SmallRng::seed_from_u64(seed);
            let record = play_game_shuffle(p0.as_mut(), p1.as_mut(), &mut rng);
            if let Some(p) = progress {
                p.fetch_add(1, Ordering::Relaxed);
            }
            (i, record)
        })
        .collect()
}

/// Run paired deals: for each deal seed, play both (P0 vs P1) and (P1 vs P0).
/// Returns (p0_wins, total_games).
pub fn run_paired_deals(
    factory0: &PlayerFactory,
    factory1: &PlayerFactory,
    n_deals: u32,
    base_seed: u64,
    progress: Option<Arc<AtomicU32>>,
) -> (u32, u32) {
    let results: Vec<u32> = (0..n_deals)
        .into_par_iter()
        .map(|i| {
            let deal_seed = base_seed.wrapping_add(i as u64);
            let mut wins = 0u32;

            // Swap 0: P0 plays as player 0
            {
                let mut p0 = factory0(deal_seed);
                let mut p1 = factory1(deal_seed ^ 0xCAFE_BABE);
                let mut rng = SmallRng::seed_from_u64(deal_seed);
                let record = play_game_shuffle(p0.as_mut(), p1.as_mut(), &mut rng);
                if record.winner() == 0 { wins += 1; }
            }

            // Swap 1: P0 plays as player 1 (same deal)
            {
                let mut p1 = factory0(deal_seed ^ 0x1111_2222);
                let mut p0 = factory1(deal_seed ^ 0x3333_4444);
                let mut rng = SmallRng::seed_from_u64(deal_seed);
                let record = play_game_shuffle(p0.as_mut(), p1.as_mut(), &mut rng);
                if record.winner() == 1 { wins += 1; }
            }

            if let Some(ref p) = progress {
                p.fetch_add(1, Ordering::Relaxed);
            }
            wins
        })
        .collect();

    let p0_wins: u32 = results.iter().sum();
    (p0_wins, n_deals * 2)
}
