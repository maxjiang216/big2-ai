use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::greedy::evaluators::{LinearEvaluator, TreeEvaluator};
use crate::greedy::{
    GreedyNoBombStrategy, GreedyPassStrategy, GreedyRandomPassStrategy, GreedyRandomStrategy,
    GreedyStrategy, GreedyLinearStrategy, TreeGreedyStrategy,
};
use crate::pimc::{
    PimcGreedyStrategy, PimcLinearStrategy, PimcPassRolloutStrategy, PimcTreeStrategy,
};
use crate::player::{AnyPlayer, Player};
use crate::random::RandomStrategy;

fn rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

/// Construct a boxed player by name.
///
/// `param` is a floating-point hyperparameter:
///   - For greedy_random / greedy_random_pass / greedy_no_bomb: probability in [0,1].
///   - For pimc* variants: number of determinizations (truncated to u32).
///   - For model-based players: the value is ignored (model path not yet configurable here).
///
/// Model paths use the default locations under `data/`.
pub fn make_player(name: &str, param: f64, seed: u64) -> Box<dyn AnyPlayer> {
    match name {
        "random" => Box::new(Player::new(RandomStrategy::new(rng(seed)))),

        "greedy" => Box::new(Player::new(GreedyStrategy)),

        "greedy_random" => {
            Box::new(Player::new(GreedyRandomStrategy::new(param, rng(seed))))
        }

        "greedy_random_pass" => {
            Box::new(Player::new(GreedyRandomPassStrategy::new(param, rng(seed))))
        }

        "greedy_no_bomb" => {
            Box::new(Player::new(GreedyNoBombStrategy::new(param, rng(seed))))
        }

        "greedy_linear" => {
            let eval = LinearEvaluator::load("data/linear_w.txt")
                .expect("greedy_linear: model not found");
            Box::new(Player::new(GreedyLinearStrategy::new(eval)))
        }

        "greedy_pass" => {
            let model = LinearEvaluator::load("data/pass_ridge_w.txt")
                .expect("greedy_pass: model not found");
            Box::new(Player::new(GreedyPassStrategy::new(model)))
        }

        "tree_greedy" => {
            let eval = TreeEvaluator::load("data/tree_model.txt")
                .expect("tree_greedy: model not found");
            Box::new(Player::new(TreeGreedyStrategy::new(eval)))
        }

        "pimc" => {
            let n = param as u32;
            Box::new(Player::new(PimcGreedyStrategy::new(n, rng(seed), false)))
        }

        "pimc_redet" => {
            let n = param as u32;
            Box::new(Player::new(PimcGreedyStrategy::new(n, rng(seed), true)))
        }

        "pimc_adaptive" => {
            // param = n_max; n_min = 10, delta = 0.05
            let n_max = param as u32;
            Box::new(Player::new(PimcGreedyStrategy::new_adaptive(n_max, 10, 0.05, rng(seed), false)))
        }

        "pimc_linear" => {
            let n = param as u32;
            let eval = LinearEvaluator::load("data/linear_w.txt")
                .expect("pimc_linear: model not found");
            Box::new(Player::new(PimcLinearStrategy::new(n, eval, rng(seed), false)))
        }

        "pimc_linear_redet" => {
            let n = param as u32;
            let eval = LinearEvaluator::load("data/linear_w.txt")
                .expect("pimc_linear_redet: model not found");
            Box::new(Player::new(PimcLinearStrategy::new(n, eval, rng(seed), true)))
        }

        "pimc_tree" => {
            let n = param as u32;
            let eval = TreeEvaluator::load("data/tree_model.txt")
                .expect("pimc_tree: model not found");
            Box::new(Player::new(PimcTreeStrategy::new(n, eval, rng(seed), false)))
        }

        "pimc_tree_redet" => {
            let n = param as u32;
            let eval = TreeEvaluator::load("data/tree_model.txt")
                .expect("pimc_tree_redet: model not found");
            Box::new(Player::new(PimcTreeStrategy::new(n, eval, rng(seed), true)))
        }

        "pimc_tree_adaptive" => {
            let n_max = param as u32;
            let eval = TreeEvaluator::load("data/tree_model.txt")
                .expect("pimc_tree_adaptive: model not found");
            Box::new(Player::new(PimcTreeStrategy::new_adaptive(n_max, 10, 0.05, eval, rng(seed), false)))
        }

        "pimc_pass_rollout" => {
            let n = param as u32;
            let model = LinearEvaluator::load("data/pass_ridge_w.txt")
                .expect("pimc_pass_rollout: model not found");
            Box::new(Player::new(PimcPassRolloutStrategy::new(n, model, rng(seed), false)))
        }

        other => panic!("make_player: unknown player name '{}'", other),
    }
}
