use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::time::Instant;

use big2_players::make_player;
use big2_simulation::coordinator::{run_paired_deals, PlayerFactory};

struct WilsonCI { lo: f64, hi: f64 }

fn wilson_ci(wins: u32, n: u32) -> WilsonCI {
    let z = 1.96f64;
    let p = wins as f64 / n as f64;
    let z2 = z * z;
    let ni = 1.0 / n as f64;
    let center = (p + z2 * ni / 2.0) / (1.0 + z2 * ni);
    let half = z * (p * (1.0 - p) * ni + z2 * ni * ni / 4.0).sqrt() / (1.0 + z2 * ni);
    WilsonCI { lo: center - half, hi: center + half }
}

fn print_usage(prog: &str) {
    eprintln!(
        "Usage: {prog} --p0 <name> --p1 <name> --deals <N> [options]

Options:
  --p0 <name>        Player A type (required)
  --p0-param <f>     Parameter for player A (default: 0.0)
  --p1 <name>        Player B type (required)
  --p1-param <f>     Parameter for player B (default: 0.0)
  --deals <N>        Number of unique deals; total games = 2*N (required)
  --seed <S>         Base RNG seed (default: 42)

Example:
  {prog} --p0 greedy --p1 random --deals 10000 --seed 42"
    );
}

fn player_label(name: &str, param: f64) -> String {
    match name {
        "greedy" | "random" | "greedy_pass" => name.to_owned(),
        _ => format!("{name}({param:.2})")
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prog = &args[0];

    let mut p0_name = String::new();
    let mut p1_name = String::new();
    let mut p0_param = 0.0f64;
    let mut p1_param = 0.0f64;
    let mut num_deals: u32 = 0;
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => { print_usage(prog); return; }
            "--p0"       => { i += 1; p0_name = args[i].clone(); }
            "--p0-param" => { i += 1; p0_param = args[i].parse().expect("float"); }
            "--p1"       => { i += 1; p1_name = args[i].clone(); }
            "--p1-param" => { i += 1; p1_param = args[i].parse().expect("float"); }
            "--deals"    => { i += 1; num_deals = args[i].parse().expect("integer"); }
            "--seed"     => { i += 1; seed = args[i].parse().expect("integer"); }
            other => { eprintln!("Unknown argument: {other}"); print_usage(prog); std::process::exit(1); }
        }
        i += 1;
    }

    if p0_name.is_empty() || p1_name.is_empty() || num_deals == 0 {
        eprintln!("Error: --p0, --p1, and --deals are required");
        print_usage(prog);
        std::process::exit(1);
    }

    let label0 = player_label(&p0_name, p0_param);
    let label1 = player_label(&p1_name, p1_param);
    let total_games = num_deals * 2;

    eprintln!("\n=== Eval Match: {label0} vs {label1} ===");
    eprintln!("Deals:   {num_deals}  |  Total games: {total_games}");
    eprintln!("Seed:    {seed}\n");

    let n0 = p0_name.clone(); let n1 = p1_name.clone();
    let factory0: PlayerFactory = Box::new(move |s| make_player(&n0, p0_param, s));
    let factory1: PlayerFactory = Box::new(move |s| make_player(&n1, p1_param, s));

    let progress = Arc::new(AtomicU32::new(0));
    let prog_clone = progress.clone();
    let n = num_deals;
    let t_start = Instant::now();

    let progress_thread = std::thread::spawn(move || {
        loop {
            let done = prog_clone.load(std::sync::atomic::Ordering::Relaxed);
            let elapsed = t_start.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { 2.0 * done as f64 / elapsed } else { 0.0 };
            let pct = if n > 0 { 100 * done / n } else { 0 };
            eprint!("\rDeals: {done}/{n} ({pct}%)  |  {rate:.0} games/s   ");
            if done >= n { break; }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    });

    let (p0_wins, total) = run_paired_deals(&factory0, &factory1, num_deals, seed, Some(progress.clone()));

    progress.store(num_deals, std::sync::atomic::Ordering::Relaxed);
    let _ = progress_thread.join();
    eprintln!();

    let elapsed = t_start.elapsed().as_secs_f64();
    let games_per_sec = total as f64 / elapsed;

    let p_hat = p0_wins as f64 / total as f64;
    let ci = wilson_ci(p0_wins, total);

    println!("P0 ({label0}) wins: {p0_wins} / {total} ({:.2}%)", 100.0 * p_hat);
    println!("95% Wilson CI: [{:.2}%, {:.2}%]", 100.0 * ci.lo, 100.0 * ci.hi);

    if ci.lo > 0.50 {
        println!("Result: {label0} is significantly better (CI excludes 50%)");
    } else if ci.hi < 0.50 {
        println!("Result: {label1} is significantly better (CI excludes 50%)");
    } else {
        println!("Result: no significant difference (CI includes 50%)");
    }

    println!("Elapsed: {elapsed:.1}s  ({games_per_sec:.0} games/s)");
}
