use big2_core::{
    consts::max_copies_in_deck, Game, Move, PartialGame, peek_tablebase_move, PASS,
};
use rand::rngs::SmallRng;
use rand::Rng;

use crate::player::Strategy;
use crate::greedy::evaluators::{
    greedy_best, greedy_hand_eval, voluntary_pass_legal, LinearEvaluator, TreeEvaluator,
};

// ---------------------------------------------------------------------------
// Hand sampler
// ---------------------------------------------------------------------------

/// Draw a random opponent hand of size `opp_count` from unseen cards
/// (not in `my_hand`, not in `discard`).
pub fn sample_opponent_hand(
    my_hand: &[u8; 13],
    discard: &[u8; 13],
    opp_count: u8,
    rng: &mut SmallRng,
) -> [u8; 13] {
    let mut pool: Vec<usize> = Vec::with_capacity(16);
    for r in 0..13usize {
        let available = max_copies_in_deck(r) - my_hand[r] - discard[r];
        for _ in 0..available {
            pool.push(r);
        }
    }

    let mut opp_hand = [0u8; 13];
    let pool_size = pool.len();
    let count = (opp_count as usize).min(pool_size);
    for i in 0..count {
        let j = rng.gen_range(i..pool_size);
        pool.swap(i, j);
        opp_hand[pool[i]] += 1;
    }
    opp_hand
}

// ---------------------------------------------------------------------------
// Hoeffding early stop
// ---------------------------------------------------------------------------

/// Returns true when a unique leader has a statistically significantly
/// higher win rate than all others (Bonferroni-corrected Hoeffding bound).
pub fn confident_leader_hoeffding(wins: &[u32], n: u32, delta: f64) -> bool {
    let m = wins.len();
    if n < 1 || m < 2 {
        return false;
    }

    let max_w = *wins.iter().max().unwrap();
    let mut leader = None;
    let mut leader_count = 0usize;
    for (i, &w) in wins.iter().enumerate() {
        if w == max_w {
            leader_count += 1;
            leader = Some(i);
        }
    }
    if leader_count != 1 { return false; }
    let b = leader.unwrap();

    let delta_prime = delta / m as f64;
    if delta_prime <= 0.0 || delta_prime >= 1.0 { return false; }

    let r = (f64::ln(2.0 / delta_prime) / (2.0 * n as f64)).sqrt();
    let inv_n = 1.0 / n as f64;

    let max_upper_other = wins.iter().enumerate()
        .filter(|&(i, _)| i != b)
        .map(|(_, &w)| w as f64 * inv_n + r)
        .fold(f64::NEG_INFINITY, f64::max);

    let lower_b = wins[b] as f64 * inv_n - r;
    lower_b > max_upper_other
}

// ---------------------------------------------------------------------------
// Rollout helpers
// ---------------------------------------------------------------------------

fn resample_game(g: &Game, my_player: usize, rng: &mut SmallRng) -> Game {
    let my_h = g.player_hand(my_player);
    let disc = g.discard_pile();
    let opp = 1 - my_player;
    let opp_count = g.hand_size(opp);
    let new_opp = sample_opponent_hand(&my_h, &disc, opp_count, rng);
    let (h0, h1) = if opp == 0 {
        (new_opp, g.player_hand(1))
    } else {
        (g.player_hand(0), new_opp)
    };
    Game::from_state(h0, h1, disc, g.last_move(), g.current_player())
}

/// Greedy + tablebase rollout to game end. Returns 1 if `my_player` wins.
pub fn greedy_rollout(mut g: Game, my_player: usize, resample: bool, rng: &mut SmallRng) -> u32 {
    while !g.is_over() {
        let cp = g.current_player();
        if resample && cp != my_player {
            g = resample_game(&g, my_player, rng);
        }
        let pg = PartialGame::from_game(&g, cp);
        let tb = peek_tablebase_move(&pg);
        let mv = if let Some(m) = tb.mv { m } else { greedy_best(&pg, greedy_hand_eval) };
        g.apply_move(mv);
    }
    if g.winner() == my_player { 1 } else { 0 }
}

/// Greedy + tablebase rollout with an ML evaluator for move selection.
pub fn ml_rollout<F>(
    mut g: Game,
    my_player: usize,
    eval_fn: &F,
    resample: bool,
    rng: &mut SmallRng,
) -> u32
where
    F: Fn(&PartialGame) -> f64,
{
    while !g.is_over() {
        let cp = g.current_player();
        if resample && cp != my_player {
            g = resample_game(&g, my_player, rng);
        }
        let pg = PartialGame::from_game(&g, cp);
        let tb = peek_tablebase_move(&pg);
        let mv = if let Some(m) = tb.mv { m } else { greedy_best(&pg, eval_fn) };
        g.apply_move(mv);
    }
    if g.winner() == my_player { 1 } else { 0 }
}

/// Like `greedy_rollout` but respects a voluntary pass model during rollout.
pub fn pass_rollout(
    mut g: Game,
    my_player: usize,
    pass_model: &LinearEvaluator,
    resample: bool,
    rng: &mut SmallRng,
) -> u32 {
    while !g.is_over() {
        let cp = g.current_player();
        if resample && cp != my_player {
            g = resample_game(&g, my_player, rng);
        }
        let pg = PartialGame::from_game(&g, cp);
        let tb = peek_tablebase_move(&pg);
        let mv = if let Some(m) = tb.mv {
            m
        } else {
            let legal = pg.legal_moves();
            if voluntary_pass_legal(&legal) && pass_model.predict(&pg) > 0.0 {
                Move::PASS
            } else {
                greedy_best(&pg, greedy_hand_eval)
            }
        };
        g.apply_move(mv);
    }
    if g.winner() == my_player { 1 } else { 0 }
}

// ---------------------------------------------------------------------------
// Core PIMC selection
// ---------------------------------------------------------------------------

/// Shared PIMC move selection with common random numbers (CRN).
///
/// `rollout` captures its evaluator; signature: (Game, my_player, resample, rng) -> u32.
pub fn pimc_select_move(
    state: &PartialGame,
    player_num: usize,
    n_max: u32,
    n_min: u32,
    adaptive: bool,
    delta: f64,
    resample: bool,
    rng: &mut SmallRng,
    rollout: &mut impl FnMut(Game, usize, bool, &mut SmallRng) -> u32,
) -> Move {
    let legal = state.legal_moves();
    let mut candidates: Vec<usize> = legal.iter().copied().filter(|&m| m != PASS).collect();
    if candidates.is_empty() { return Move::PASS; }
    let has_pass = legal.contains(&PASS);
    if has_pass { candidates.push(PASS); }

    let my_hand = state.player_hand();
    let discard = state.discard_pile();
    let opp_count = state.opponent_hand_size();
    let last_mv = state.last_move();

    let m = candidates.len();
    let mut wins = vec![0u32; m];

    for det in 0..n_max {
        let opp_hand = sample_opponent_hand(&my_hand, &discard, opp_count, rng);
        let (h0, h1) = if player_num == 0 { (my_hand, opp_hand) } else { (opp_hand, my_hand) };

        for (ci, &cand) in candidates.iter().enumerate() {
            let mut g = Game::from_state(h0, h1, discard, last_mv, player_num);
            g.apply_move(Move::decode(cand));
            wins[ci] += rollout(g, player_num, resample, rng);
        }

        let dets_done = det + 1;
        if adaptive && dets_done >= n_min {
            if m == 1 || confident_leader_hoeffding(&wins, dets_done, delta) {
                break;
            }
        }
    }

    // Best candidate: most wins; tiebreak by lowest move id.
    let best_ci = wins.iter().enumerate()
        .max_by(|&(ai, &aw), &(bi, &bw)| {
            aw.cmp(&bw).then(candidates[bi].cmp(&candidates[ai]))
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    Move::decode(candidates[best_ci])
}

// ---------------------------------------------------------------------------
// PimcGreedyStrategy — greedy heuristic rollout
// ---------------------------------------------------------------------------

pub struct PimcGreedyStrategy {
    n_max: u32,
    n_min: u32,
    adaptive: bool,
    delta: f64,
    resample: bool,
    player_num: usize,
    pub rng: SmallRng,
}

impl PimcGreedyStrategy {
    pub fn new(n_max: u32, rng: SmallRng, resample: bool) -> Self {
        PimcGreedyStrategy { n_max, n_min: 0, adaptive: false, delta: 0.05, resample, player_num: 0, rng }
    }

    pub fn new_adaptive(n_max: u32, n_min: u32, delta: f64, rng: SmallRng, resample: bool) -> Self {
        PimcGreedyStrategy { n_max, n_min, adaptive: true, delta, resample, player_num: 0, rng }
    }
}

impl Strategy for PimcGreedyStrategy {
    fn on_deal(&mut self, player_num: usize) { self.player_num = player_num; }

    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        pimc_select_move(
            state, self.player_num, self.n_max, self.n_min, self.adaptive,
            self.delta, self.resample, &mut self.rng,
            &mut |g, p, re, rng| greedy_rollout(g, p, re, rng),
        )
    }
}

// ---------------------------------------------------------------------------
// PimcTreeStrategy — tree evaluator rollout
// ---------------------------------------------------------------------------

pub struct PimcTreeStrategy {
    n_max: u32,
    n_min: u32,
    adaptive: bool,
    delta: f64,
    resample: bool,
    player_num: usize,
    evaluator: TreeEvaluator,
    rng: SmallRng,
}

impl PimcTreeStrategy {
    pub fn new(n_max: u32, evaluator: TreeEvaluator, rng: SmallRng, resample: bool) -> Self {
        PimcTreeStrategy { n_max, n_min: 0, adaptive: false, delta: 0.05, resample, player_num: 0, evaluator, rng }
    }

    pub fn new_adaptive(n_max: u32, n_min: u32, delta: f64, evaluator: TreeEvaluator, rng: SmallRng, resample: bool) -> Self {
        PimcTreeStrategy { n_max, n_min, adaptive: true, delta, resample, player_num: 0, evaluator, rng }
    }
}

impl Strategy for PimcTreeStrategy {
    fn on_deal(&mut self, player_num: usize) { self.player_num = player_num; }

    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        let eval = &self.evaluator;
        pimc_select_move(
            state, self.player_num, self.n_max, self.n_min, self.adaptive,
            self.delta, self.resample, &mut self.rng,
            &mut |g, p, re, rng| ml_rollout(g, p, &|s| eval.predict(s), re, rng),
        )
    }
}

// ---------------------------------------------------------------------------
// PimcLinearStrategy — Ridge linear rollout
// ---------------------------------------------------------------------------

pub struct PimcLinearStrategy {
    n_max: u32,
    n_min: u32,
    adaptive: bool,
    delta: f64,
    resample: bool,
    player_num: usize,
    evaluator: LinearEvaluator,
    rng: SmallRng,
}

impl PimcLinearStrategy {
    pub fn new(n_max: u32, evaluator: LinearEvaluator, rng: SmallRng, resample: bool) -> Self {
        PimcLinearStrategy { n_max, n_min: 0, adaptive: false, delta: 0.05, resample, player_num: 0, evaluator, rng }
    }

    pub fn new_adaptive(n_max: u32, n_min: u32, delta: f64, evaluator: LinearEvaluator, rng: SmallRng, resample: bool) -> Self {
        PimcLinearStrategy { n_max, n_min, adaptive: true, delta, resample, player_num: 0, evaluator, rng }
    }
}

impl Strategy for PimcLinearStrategy {
    fn on_deal(&mut self, player_num: usize) { self.player_num = player_num; }

    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        let eval = &self.evaluator;
        pimc_select_move(
            state, self.player_num, self.n_max, self.n_min, self.adaptive,
            self.delta, self.resample, &mut self.rng,
            &mut |g, p, re, rng| ml_rollout(g, p, &|s| eval.predict(s), re, rng),
        )
    }
}

// ---------------------------------------------------------------------------
// PimcPassRolloutStrategy — greedy PIMC with pass-aware rollout
// ---------------------------------------------------------------------------

pub struct PimcPassRolloutStrategy {
    n_max: u32,
    n_min: u32,
    adaptive: bool,
    delta: f64,
    resample: bool,
    player_num: usize,
    pass_model: LinearEvaluator,
    rng: SmallRng,
}

impl PimcPassRolloutStrategy {
    pub fn new(n_max: u32, pass_model: LinearEvaluator, rng: SmallRng, resample: bool) -> Self {
        PimcPassRolloutStrategy { n_max, n_min: 0, adaptive: false, delta: 0.05, resample, player_num: 0, pass_model, rng }
    }

    pub fn new_adaptive(n_max: u32, n_min: u32, delta: f64, pass_model: LinearEvaluator, rng: SmallRng, resample: bool) -> Self {
        PimcPassRolloutStrategy { n_max, n_min, adaptive: true, delta, resample, player_num: 0, pass_model, rng }
    }
}

impl Strategy for PimcPassRolloutStrategy {
    fn on_deal(&mut self, player_num: usize) { self.player_num = player_num; }

    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        let pass_model = &self.pass_model;
        pimc_select_move(
            state, self.player_num, self.n_max, self.n_min, self.adaptive,
            self.delta, self.resample, &mut self.rng,
            &mut |g, p, re, rng| pass_rollout(g, p, pass_model, re, rng),
        )
    }
}
