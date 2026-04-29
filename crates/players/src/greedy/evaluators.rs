use std::io::{BufRead, BufReader};
use big2_core::{Combination, Move, MOVE_TO_CARDS, PartialGame};

pub const TREE_N_FEATURES: usize = 60;

// ---------------------------------------------------------------------------
// Feature extraction — matches FEATURE_COLS in analysis/train_tree_greedy.py
// ---------------------------------------------------------------------------

fn highest_rank_with_count(hand: &[u8; 13], count: u8) -> i32 {
    for r in (0..13i32).rev() {
        if hand[r as usize] >= count {
            return r;
        }
    }
    -1
}

fn highest_rank_with_count_not_bomb(hand: &[u8; 13], count: u8) -> i32 {
    for i in (0..13i32).rev() {
        let ui = i as usize;
        let is_bomb = (ui == 11 && hand[ui] == 3) || (ui != 11 && hand[ui] == 4);
        if hand[ui] >= count && !is_bomb {
            return i;
        }
    }
    -1
}

fn count_ge(hand: &[u8; 13], start: usize) -> u32 {
    hand[start..].iter().map(|&c| c as u32).sum()
}

fn count_le(hand: &[u8; 13], end: usize) -> u32 {
    hand[..=end].iter().map(|&c| c as u32).sum()
}

pub fn extract_tree_features(state: &PartialGame) -> [f32; TREE_N_FEATURES] {
    let h = state.player_hand();
    let lm = state.last_move();
    let c = lm.combination;

    let n_cards: u32 = h.iter().map(|&x| x as u32).sum();
    let n_bombs = {
        let mut b = 0u32;
        for i in 0..13usize {
            if (i == 11 && h[i] == 3) || (i != 11 && h[i] == 4) {
                b += 1;
            }
        }
        b
    };

    let pm = state.possible_moves().len() as f32;
    let pm_nb = state.possible_moves_not_bomb().len() as f32;

    let mid = lm.encode();
    let last_card_count = MOVE_TO_CARDS[mid][13] as f32;
    let trick_rank = state.trick_rank().unwrap_or(0) as f32;

    let mut f = [0f32; TREE_N_FEATURES];
    let mut k = 0;

    f[k] = n_cards as f32; k += 1;
    f[k] = state.opponent_hand_size() as f32; k += 1;
    f[k] = if state.hand_is_only_singles() { 1.0 } else { 0.0 }; k += 1;

    for i in 0..13 {
        f[k] = h[i] as f32; k += 1;
    }

    // n_ge_4 .. n_ge_A (start indices 1..=11)
    for start in 1..=11usize {
        f[k] = count_ge(&h, start) as f32; k += 1;
    }

    // n_le_3 .. n_le_A (end indices 0..=11)
    for end in 0..=11usize {
        f[k] = count_le(&h, end) as f32; k += 1;
    }

    f[k] = highest_rank_with_count(&h, 1) as f32; k += 1;
    f[k] = highest_rank_with_count(&h, 2) as f32; k += 1;
    f[k] = highest_rank_with_count(&h, 3) as f32; k += 1;
    f[k] = highest_rank_with_count(&h, 4) as f32; k += 1;

    f[k] = highest_rank_with_count_not_bomb(&h, 1) as f32; k += 1;
    f[k] = highest_rank_with_count_not_bomb(&h, 2) as f32; k += 1;
    f[k] = highest_rank_with_count_not_bomb(&h, 3) as f32; k += 1;

    f[k] = if c == Combination::Pass      { 1.0 } else { 0.0 }; k += 1;
    f[k] = if c == Combination::Single    { 1.0 } else { 0.0 }; k += 1;
    f[k] = if c == Combination::Double    { 1.0 } else { 0.0 }; k += 1;
    f[k] = if c == Combination::Triple    { 1.0 } else { 0.0 }; k += 1;
    f[k] = if c == Combination::FullHouse { 1.0 } else { 0.0 }; k += 1;
    f[k] = if c == Combination::Bomb      { 1.0 } else { 0.0 }; k += 1;
    f[k] = if c.is_straight()             { 1.0 } else { 0.0 }; k += 1;
    f[k] = if c.is_double_straight()      { 1.0 } else { 0.0 }; k += 1;
    f[k] = if c.is_triple_straight()      { 1.0 } else { 0.0 }; k += 1;

    f[k] = last_card_count; k += 1;
    f[k] = n_bombs as f32; k += 1;
    f[k] = pm; k += 1;
    f[k] = pm_nb; k += 1;
    f[k] = trick_rank; k += 1;

    debug_assert_eq!(k, TREE_N_FEATURES);
    f
}

// ---------------------------------------------------------------------------
// TreeEvaluator — decision tree that predicts win probability.
//
// Text format:
//   line 1: n_nodes n_features
//   remaining lines: feature_idx threshold left right value
//   leaf nodes have feature_idx == -2
// ---------------------------------------------------------------------------

pub struct TreeEvaluator {
    feature: Vec<i32>,
    threshold: Vec<f32>,
    left: Vec<i32>,
    right: Vec<i32>,
    value: Vec<f32>,
}

impl TreeEvaluator {
    pub fn load(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("TreeEvaluator: cannot open '{}': {}", path, e))?;
        let mut lines = BufReader::new(file).lines();

        let header = lines.next()
            .ok_or_else(|| format!("TreeEvaluator: empty file '{}'", path))?
            .map_err(|e| e.to_string())?;
        let mut parts = header.split_whitespace();
        let n_nodes: usize = parts.next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("TreeEvaluator: bad header in '{}'", path))?;
        let n_features: usize = parts.next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("TreeEvaluator: bad header in '{}'", path))?;
        if n_features != TREE_N_FEATURES {
            return Err(format!(
                "TreeEvaluator: feature count mismatch (file={} expected={})",
                n_features, TREE_N_FEATURES
            ));
        }

        let mut feature = Vec::with_capacity(n_nodes);
        let mut threshold = Vec::with_capacity(n_nodes);
        let mut left = Vec::with_capacity(n_nodes);
        let mut right = Vec::with_capacity(n_nodes);
        let mut value = Vec::with_capacity(n_nodes);

        for (i, line) in lines.enumerate() {
            if i >= n_nodes { break; }
            let line = line.map_err(|e| e.to_string())?;
            let mut p = line.split_whitespace();
            let fi: i32 = p.next().and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("TreeEvaluator: bad node {} in '{}'", i, path))?;
            let th: f32 = p.next().and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("TreeEvaluator: bad node {} in '{}'", i, path))?;
            let l: i32 = p.next().and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("TreeEvaluator: bad node {} in '{}'", i, path))?;
            let r: i32 = p.next().and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("TreeEvaluator: bad node {} in '{}'", i, path))?;
            let v: f32 = p.next().and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("TreeEvaluator: bad node {} in '{}'", i, path))?;
            feature.push(fi);
            threshold.push(th);
            left.push(l);
            right.push(r);
            value.push(v);
        }

        Ok(TreeEvaluator { feature, threshold, left, right, value })
    }

    pub fn predict(&self, state: &PartialGame) -> f64 {
        let features = extract_tree_features(state);
        let mut node = 0usize;
        loop {
            let fi = self.feature[node];
            if fi == -2 {
                return self.value[node] as f64;
            }
            let x = features[fi as usize];
            node = if x <= self.threshold[node] {
                self.left[node] as usize
            } else {
                self.right[node] as usize
            };
        }
    }
}

// ---------------------------------------------------------------------------
// LinearEvaluator — Ridge regression on a subset of tree features.
//
// Text format (6 lines):
//   k
//   intercept
//   k feature indices
//   k means
//   k scales
//   k coefficients
// ---------------------------------------------------------------------------

pub struct LinearEvaluator {
    k: usize,
    intercept: f64,
    feat_idx: Vec<usize>,
    mean: Vec<f64>,
    scale: Vec<f64>,
    coef: Vec<f64>,
}

impl LinearEvaluator {
    pub fn load(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("LinearEvaluator: cannot open '{}': {}", path, e))?;
        let mut nums = content.split_whitespace();

        macro_rules! next_val {
            ($t:ty) => {
                nums.next()
                    .ok_or_else(|| format!("LinearEvaluator: unexpected EOF in '{}'", path))?
                    .parse::<$t>()
                    .map_err(|e| format!("LinearEvaluator: parse error in '{}': {}", path, e))?
            };
        }

        let k = next_val!(usize);
        if k == 0 || k > TREE_N_FEATURES {
            return Err(format!("LinearEvaluator: invalid k={} in '{}'", k, path));
        }
        let intercept = next_val!(f64);

        let mut feat_idx = Vec::with_capacity(k);
        for _ in 0..k {
            let idx = next_val!(usize);
            if idx >= TREE_N_FEATURES {
                return Err(format!("LinearEvaluator: feature index {} out of range in '{}'", idx, path));
            }
            feat_idx.push(idx);
        }

        let mut mean = Vec::with_capacity(k);
        for _ in 0..k { mean.push(next_val!(f64)); }

        let mut scale = Vec::with_capacity(k);
        for _ in 0..k { scale.push(next_val!(f64)); }

        let mut coef = Vec::with_capacity(k);
        for _ in 0..k { coef.push(next_val!(f64)); }

        Ok(LinearEvaluator { k, intercept, feat_idx, mean, scale, coef })
    }

    pub fn predict(&self, state: &PartialGame) -> f64 {
        let f = extract_tree_features(state);
        let mut s = self.intercept;
        for j in 0..self.k {
            let x = f[self.feat_idx[j]] as f64;
            let z = x - self.mean[j];
            let sc = if self.scale[j] < 1e-12 { 1.0 } else { self.scale[j] };
            s += self.coef[j] * z / sc;
        }
        s
    }
}

// ---------------------------------------------------------------------------
// GreedyHeuristic — the hand-crafted lexicographic evaluator.
// ---------------------------------------------------------------------------

/// Lexicographic hand evaluation — higher is better.
/// Derived PartialOrd gives free lexicographic comparison in the order declared.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct GreedyEval {
    pub win_now: i8,
    pub bombs: i8,
    pub neg_num_cards: i8,
    pub num_2s: i8,
    pub num_as: i8,
    pub num_ks: i8,
    pub num_qs: i8,
    pub num_js: i8,
    pub num_10s: i8,
    pub num_9s: i8,
    pub num_8s: i8,
    pub num_7s: i8,
    pub num_6s: i8,
    pub num_5s: i8,
    pub num_4s: i8,
}

pub fn greedy_hand_eval(state: &PartialGame) -> GreedyEval {
    let h = state.player_hand();
    let mut n_cards: i8 = 0;
    let mut n_bombs: i8 = 0;
    for i in 0..13usize {
        n_cards += h[i] as i8;
        if (i < 11 && h[i] == 4) || (i == 11 && h[i] == 3) {
            n_bombs += 1;
        }
    }
    GreedyEval {
        win_now:       if n_cards == 0 { 1 } else { 0 },
        bombs:         n_bombs,
        neg_num_cards: -n_cards,
        num_2s:        h[12] as i8,
        num_as:        h[11] as i8,
        num_ks:        h[10] as i8,
        num_qs:        h[9] as i8,
        num_js:        h[8] as i8,
        num_10s:       h[7] as i8,
        num_9s:        h[6] as i8,
        num_8s:        h[5] as i8,
        num_7s:        h[4] as i8,
        num_6s:        h[3] as i8,
        num_5s:        h[2] as i8,
        num_4s:        h[1] as i8,
    }
}

// ---------------------------------------------------------------------------
// greedy_best generic function
// ---------------------------------------------------------------------------

/// Apply each non-pass legal move to a scratch PartialGame, evaluate with
/// `eval_fn(post_move_state)`, and return the move with the highest score.
/// Falls back to PASS if there are no non-pass moves.
pub fn greedy_best<F, E>(state: &PartialGame, eval_fn: F) -> Move
where
    F: Fn(&PartialGame) -> E,
    E: PartialOrd,
{
    use big2_core::PASS;
    let legal = state.legal_moves();
    let non_pass: Vec<usize> = legal.into_iter().filter(|&m| m != PASS).collect();
    if non_pass.is_empty() {
        return Move::PASS;
    }

    let mut best_idx = 0;
    let mut sim = state.clone();
    sim.apply_move(Move::decode(non_pass[0]));
    let mut best_val = eval_fn(&sim);

    for &mid in &non_pass[1..] {
        let mut sim = state.clone();
        sim.apply_move(Move::decode(mid));
        let val = eval_fn(&sim);
        if val > best_val {
            best_val = val;
            best_idx = non_pass.iter().position(|&m| m == mid).unwrap();
        }
    }
    Move::decode(non_pass[best_idx])
}

/// True iff pass is legal and at least one non-pass move is also legal.
pub fn voluntary_pass_legal(legal: &[usize]) -> bool {
    use big2_core::PASS;
    let has_pass = legal.contains(&PASS);
    let has_non = legal.iter().any(|&m| m != PASS);
    has_pass && has_non
}

#[cfg(test)]
mod tests {
    use super::*;
    use big2_core::{Game, PartialGame};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_extract_tree_features_len() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut g = Game::new();
        g.shuffle_deal(&mut rng);
        let pg = PartialGame::from_game(&g, 0);
        let f = extract_tree_features(&pg);
        assert_eq!(f.len(), TREE_N_FEATURES);
    }

    #[test]
    fn test_greedy_eval_order() {
        // win_now=1 beats win_now=0
        let a = GreedyEval { win_now: 1, bombs: 0, neg_num_cards: 0, num_2s: 0, num_as: 0,
            num_ks: 0, num_qs: 0, num_js: 0, num_10s: 0, num_9s: 0,
            num_8s: 0, num_7s: 0, num_6s: 0, num_5s: 0, num_4s: 0 };
        let b = GreedyEval { win_now: 0, ..a };
        assert!(a > b);
    }
}
