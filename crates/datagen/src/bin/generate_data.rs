use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use big2_datagen::parquet::export_parquet;
use big2_features::registry::{create_feature, Feature};
use big2_players::make_player;
use big2_simulation::coordinator::{run_games_parallel, PlayerFactory};

fn print_usage(prog: &str) {
    eprintln!(
        "Usage: {prog} --player <name> --games <N> [options]

Options:
  --player <name>          Player type for both sides (required)
  --games <N>              Number of games (required)
  --output <path>          Write <path>_game.parquet and <path>_turn.parquet
  --game-features <f1,f2>  Comma-separated game-level feature names
  --turn-features <f1,f2>  Comma-separated turn-level feature names
  --seed <S>               RNG seed (default: 42)
  --player-param <f>       Extra parameter for the player (default: 0.0)

Available features (game-level):
  outcome, length, start_legal_moves, tb_hits, tb_case1, tb_case2,
  tb_forced_seq_len, tb_forced_seq_len_sum, tb_opp1_table_straight

Available features (turn-level):
  turn_outcome, next_player, player_hand_size, opponent_hand_size,
  n_3..n_2, n_ge_4..n_ge_2, n_le_3..n_le_a, highest_single/double/triple/bomb,
  highest_single/double/triple_not_bomb, last_move_is_pass/single/double/triple/
  full_house/bomb/single_straight/double_straight/triple_straight,
  last_move_card_count, n_bombs, possible_moves, possible_moves_not_bomb,
  trick_rank, tb_case, only_single"
    );
}

fn split_csv(s: &str) -> Vec<String> {
    s.split(',').filter(|t| !t.is_empty()).map(str::to_owned).collect()
}

fn format_eta(secs: f64) -> String {
    if secs < 0.0 || !secs.is_finite() { return "?".to_owned(); }
    let s = secs.round() as u64;
    if s < 60 { return format!("{}s", s); }
    let m = s / 60; let s = s % 60;
    if m < 60 { return format!("{}m {}s", m, s); }
    let h = m / 60; let m = m % 60;
    format!("{}h {}m", h, m)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prog = &args[0];

    let mut player_name = String::new();
    let mut player_param = 0.0f64;
    let mut num_games: u32 = 0;
    let mut output_path = String::new();
    let mut game_feat_names: Vec<String> = Vec::new();
    let mut turn_feat_names: Vec<String> = Vec::new();
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => { print_usage(prog); return; }
            "--player"        => { i += 1; player_name = args[i].clone(); }
            "--games"         => { i += 1; num_games = args[i].parse().expect("--games: integer"); }
            "--output"        => { i += 1; output_path = args[i].clone(); }
            "--game-features" => { i += 1; game_feat_names = split_csv(&args[i]); }
            "--turn-features" => { i += 1; turn_feat_names = split_csv(&args[i]); }
            "--seed"          => { i += 1; seed = args[i].parse().expect("--seed: integer"); }
            "--player-param"  => { i += 1; player_param = args[i].parse().expect("--player-param: float"); }
            other => { eprintln!("Unknown argument: {other}"); print_usage(prog); std::process::exit(1); }
        }
        i += 1;
    }

    if player_name.is_empty() || num_games == 0 {
        eprintln!("Error: --player and --games are required");
        print_usage(prog);
        std::process::exit(1);
    }

    // Resolve features
    let mut game_feats = Vec::new();
    let mut turn_feats = Vec::new();
    for n in &game_feat_names {
        match create_feature(n) {
            Some(Feature::Game(f)) => { eprintln!("Game feature: {}", n); game_feats.push(f); }
            Some(Feature::Turn(_)) => eprintln!("Warning: '{}' is a turn-level feature, ignoring in --game-features", n),
            None => {}
        }
    }
    for n in &turn_feat_names {
        match create_feature(n) {
            Some(Feature::Turn(f)) => { eprintln!("Turn feature: {}", n); turn_feats.push(f); }
            Some(Feature::Game(_)) => eprintln!("Warning: '{}' is a game-level feature, ignoring in --turn-features", n),
            None => {}
        }
    }

    eprintln!("\n=== Data generation / self-play ===");
    eprintln!("Player:   {player_name} vs {player_name}");
    eprintln!("Param:    {player_param}");
    eprintln!("Games:    {num_games}");
    if !output_path.is_empty() {
        eprintln!("Output:   {output_path}_game.parquet");
        eprintln!("          {output_path}_turn.parquet");
    } else {
        eprintln!("Output:   (none — stats only)");
    }
    eprintln!("Seed:     {seed}\n");

    let p = player_param;
    let name0 = player_name.clone();
    let name1 = player_name.clone();
    let factory0: PlayerFactory = Box::new(move |s| make_player(&name0, p, s));
    let factory1: PlayerFactory = Box::new(move |s| make_player(&name1, p, s ^ 0xABCD_1234));

    let progress = Arc::new(AtomicU32::new(0));
    let prog_clone = progress.clone();
    let n = num_games;
    let t_start = Instant::now();

    // Progress printer thread
    let progress_thread = std::thread::spawn(move || {
        loop {
            let done = prog_clone.load(Ordering::Relaxed);
            let elapsed = t_start.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { done as f64 / elapsed } else { 0.0 };
            let eta = if done > 0 && rate > 0.0 { format_eta((n - done) as f64 / rate) } else { "?".to_owned() };
            let pct = if n > 0 { 100 * done / n } else { 0 };
            eprint!("\rPlaying games: {done}/{n} ({pct}%) | {rate:.1} games/s | ETA {eta}   ");
            if done >= n { break; }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    });

    let records = run_games_parallel(&factory0, &factory1, num_games, seed, Some(&progress));

    progress.store(num_games, Ordering::Relaxed);
    let _ = progress_thread.join();
    eprintln!();

    eprintln!("Elapsed: {:.1}s", t_start.elapsed().as_secs_f64());

    if !output_path.is_empty() {
        let game_path = format!("{output_path}_game.parquet");
        let turn_path = format!("{output_path}_turn.parquet");
        if let Err(e) = export_parquet(&records, &game_path, &turn_path, &game_feats, &turn_feats) {
            eprintln!("Export error: {e}");
            std::process::exit(1);
        }
    }

    eprintln!("Done.");
}
