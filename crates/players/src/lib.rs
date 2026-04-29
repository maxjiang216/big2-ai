pub mod player;
pub mod random;
pub mod greedy;
pub mod pimc;
pub mod registry;

pub use player::{AnyPlayer, Player, Strategy};
pub use random::RandomStrategy;
pub use greedy::{
    GreedyStrategy, GreedyRandomStrategy, GreedyRandomPassStrategy,
    GreedyNoBombStrategy, GreedyLinearStrategy, GreedyPassStrategy, TreeGreedyStrategy,
};
pub use greedy::evaluators::{
    GreedyEval, greedy_hand_eval, greedy_best, voluntary_pass_legal,
    extract_tree_features, TreeEvaluator, LinearEvaluator, TREE_N_FEATURES,
};
pub use pimc::{
    PimcGreedyStrategy, PimcTreeStrategy, PimcLinearStrategy, PimcPassRolloutStrategy,
    sample_opponent_hand, confident_leader_hoeffding,
};
pub use registry::make_player;
