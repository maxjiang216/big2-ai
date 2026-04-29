pub mod evaluators;

use big2_core::{Combination, Move, PartialGame, PASS};
use rand::rngs::SmallRng;
use rand::Rng;

use crate::player::Strategy;
use evaluators::{greedy_best, greedy_hand_eval, voluntary_pass_legal, LinearEvaluator, TreeEvaluator};

// ---------------------------------------------------------------------------
// GreedyStrategy — pure lexicographic heuristic
// ---------------------------------------------------------------------------

pub struct GreedyStrategy;

impl Strategy for GreedyStrategy {
    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        greedy_best(state, greedy_hand_eval)
    }
}

// ---------------------------------------------------------------------------
// GreedyRandomStrategy — with probability p, play a uniformly random legal move
// ---------------------------------------------------------------------------

pub struct GreedyRandomStrategy {
    p: f64,
    rng: SmallRng,
}

impl GreedyRandomStrategy {
    pub fn new(p: f64, rng: SmallRng) -> Self {
        GreedyRandomStrategy { p, rng }
    }
}

impl Strategy for GreedyRandomStrategy {
    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        let legal = state.legal_moves();
        if self.rng.gen::<f64>() < self.p {
            let idx = self.rng.gen_range(0..legal.len());
            return Move::decode(legal[idx]);
        }
        greedy_best(state, greedy_hand_eval)
    }
}

// ---------------------------------------------------------------------------
// GreedyRandomPassStrategy — with probability p, voluntarily pass (if possible)
// ---------------------------------------------------------------------------

pub struct GreedyRandomPassStrategy {
    p: f64,
    rng: SmallRng,
}

impl GreedyRandomPassStrategy {
    pub fn new(p: f64, rng: SmallRng) -> Self {
        GreedyRandomPassStrategy { p, rng }
    }
}

impl Strategy for GreedyRandomPassStrategy {
    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        let legal = state.legal_moves();
        let pass_available = legal.contains(&PASS);
        if pass_available && self.rng.gen::<f64>() < self.p {
            return Move::PASS;
        }
        greedy_best(state, greedy_hand_eval)
    }
}

// ---------------------------------------------------------------------------
// GreedyNoBombStrategy — suppresses bombs with probability (1 - p_bomb)
// ---------------------------------------------------------------------------

pub struct GreedyNoBombStrategy {
    p_bomb: f64,
    rng: SmallRng,
}

impl GreedyNoBombStrategy {
    pub fn new(p_bomb: f64, rng: SmallRng) -> Self {
        GreedyNoBombStrategy { p_bomb, rng }
    }
}

impl Strategy for GreedyNoBombStrategy {
    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        let legal = state.legal_moves();
        let chosen = greedy_best(state, greedy_hand_eval);
        if chosen.combination != Combination::Bomb {
            return chosen;
        }
        if self.rng.gen::<f64>() < self.p_bomb {
            return chosen;
        }
        // Suppress bomb: prefer pass, then best non-bomb, then bomb as fallback.
        if legal.contains(&PASS) {
            return Move::PASS;
        }
        let no_bomb: Vec<usize> = legal.into_iter()
            .filter(|&m| Move::decode(m).combination != Combination::Bomb)
            .collect();
        if !no_bomb.is_empty() {
            return greedy_best_from_list(state, &no_bomb);
        }
        chosen
    }
}

fn greedy_best_from_list(state: &PartialGame, moves: &[usize]) -> Move {
    if moves.is_empty() {
        return Move::PASS;
    }
    let mut best_idx = 0;
    let mut sim = state.clone();
    sim.apply_move(Move::decode(moves[0]));
    let mut best_val = greedy_hand_eval(&sim);
    for (i, &mid) in moves[1..].iter().enumerate() {
        let mut sim = state.clone();
        sim.apply_move(Move::decode(mid));
        let val = greedy_hand_eval(&sim);
        if val > best_val {
            best_val = val;
            best_idx = i + 1;
        }
    }
    Move::decode(moves[best_idx])
}

// ---------------------------------------------------------------------------
// GreedyLinearStrategy — greedy with Ridge linear evaluator
// ---------------------------------------------------------------------------

pub struct GreedyLinearStrategy {
    evaluator: LinearEvaluator,
}

impl GreedyLinearStrategy {
    pub fn new(evaluator: LinearEvaluator) -> Self {
        GreedyLinearStrategy { evaluator }
    }
}

impl Strategy for GreedyLinearStrategy {
    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        let eval = &self.evaluator;
        greedy_best(state, |s| eval.predict(s))
    }
}

// ---------------------------------------------------------------------------
// GreedyPassStrategy — greedy with optional voluntary pass when Ridge says Δ>0
// ---------------------------------------------------------------------------

pub struct GreedyPassStrategy {
    pass_model: LinearEvaluator,
}

impl GreedyPassStrategy {
    pub fn new(pass_model: LinearEvaluator) -> Self {
        GreedyPassStrategy { pass_model }
    }
}

impl Strategy for GreedyPassStrategy {
    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        let legal = state.legal_moves();
        if voluntary_pass_legal(&legal) && self.pass_model.predict(state) > 0.0 {
            return Move::PASS;
        }
        greedy_best(state, greedy_hand_eval)
    }
}

// ---------------------------------------------------------------------------
// TreeGreedyStrategy — greedy with decision tree evaluator
// ---------------------------------------------------------------------------

pub struct TreeGreedyStrategy {
    evaluator: TreeEvaluator,
}

impl TreeGreedyStrategy {
    pub fn new(evaluator: TreeEvaluator) -> Self {
        TreeGreedyStrategy { evaluator }
    }
}

impl Strategy for TreeGreedyStrategy {
    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        let eval = &self.evaluator;
        greedy_best(state, |s| eval.predict(s))
    }
}
