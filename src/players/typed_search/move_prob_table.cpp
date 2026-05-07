#include "move_prob_table.h"

#include "move.h"
#include "util.h"

#include <cstdint>
#include <fstream>

namespace typed_search {

namespace {
// Bumped to v3: factored per-component representation.
constexpr char kMagicMain[4] = {'M', 'P', 'B', '3'};
constexpr char kMagicFallback[4] = {'M', 'P', 'F', '3'};

inline int rank_idx_of_face(int face) {
  return (face == 2 || face == 15) ? 12 : (face - 3);
}

// Bomb aux index: 0 = bare bomb, 1..13 = rank_idx_of(aux_face) + 1.
inline int bomb_aux_idx_of(int aux_face) {
  if (aux_face == 0) return 0;  // bare
  return rank_idx_of_face(aux_face) + 1;
}
}  // namespace

float move_prob_weight_of(int response_id, int player_move,
                            const MoveProbComponents &cv) {
  const Move &m = all_moves()[response_id];
  if (m.combination == Move::Combination::kPass) return cv.pass;
  if (m.combination == Move::Combination::kBomb) {
    int R = rank_idx_of_face(m.rank);
    int A = bomb_aux_idx_of(m.auxiliary);
    if (R < 0 || R >= MoveProbComponents::kRanks) return 0.0f;
    if (A < 0 || A >= MoveProbComponents::kAuxRanks) return 0.0f;
    return cv.bomb_rank[R] * cv.bomb_aux[A];
  }
  // Same-type higher response. Uses main_rank.
  int R = rank_idx_of_face(m.rank);
  if (R < 0 || R >= MoveProbComponents::kRanks) return 0.0f;
  float w = cv.main_rank[R];
  if (m.combination == Move::Combination::kFullHouse) {
    int A = rank_idx_of_face(m.auxiliary);
    if (A >= 0 && A < MoveProbComponents::kRanks) {
      w *= cv.fh_aux[A];
    }
  }
  // For non-FH non-bomb non-PASS responses (singles/doubles/triples/straights),
  // the player_move's combination is matched and rank determines the response.
  // We don't condition on player_move here since the caller's legal set is
  // already filtered to valid responses.
  (void)player_move;
  return w;
}

void MoveProbTable::load(const std::string &path) {
  entries_.clear();
  std::ifstream in(path, std::ios::binary);
  if (!in) return;

  char magic[4];
  in.read(magic, 4);
  if (in.gcount() != 4) return;
  const char (&want)[4] = ignore_opp_count_ ? kMagicFallback : kMagicMain;
  if (magic[0] != want[0] || magic[1] != want[1] || magic[2] != want[2] ||
      magic[3] != want[3]) {
    return;
  }
  std::uint32_t n = 0;
  in.read(reinterpret_cast<char *>(&n), 4);
  if (!in) return;

  entries_.reserve(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint16_t player_move = 0;
    std::uint8_t opp_count = 0, our_b = 0;
    in.read(reinterpret_cast<char *>(&player_move), 2);
    in.read(reinterpret_cast<char *>(&opp_count), 1);
    in.read(reinterpret_cast<char *>(&our_b), 1);
    Entry e;
    in.read(reinterpret_cast<char *>(&e.pass_count), 4);
    in.read(reinterpret_cast<char *>(&e.pass_trial), 4);
    in.read(reinterpret_cast<char *>(e.main_count.data()), sizeof(float) * kRanks);
    in.read(reinterpret_cast<char *>(e.main_trial.data()), sizeof(float) * kRanks);
    in.read(reinterpret_cast<char *>(e.bomb_rank_count.data()), sizeof(float) * kRanks);
    in.read(reinterpret_cast<char *>(e.bomb_rank_trial.data()), sizeof(float) * kRanks);
    in.read(reinterpret_cast<char *>(e.bomb_aux_count.data()), sizeof(float) * kAuxRanks);
    in.read(reinterpret_cast<char *>(e.bomb_aux_trial.data()), sizeof(float) * kAuxRanks);
    in.read(reinterpret_cast<char *>(e.fh_aux_count.data()), sizeof(float) * kRanks);
    in.read(reinterpret_cast<char *>(e.fh_aux_trial.data()), sizeof(float) * kRanks);
    if (!in) break;
    std::uint32_t key = make_key(static_cast<int>(player_move),
                                  static_cast<int>(opp_count),
                                  static_cast<int>(our_b));
    entries_[key] = std::move(e);
  }
}

void MoveProbTable::save(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  const char (&want)[4] = ignore_opp_count_ ? kMagicFallback : kMagicMain;
  out.write(want, 4);
  std::uint32_t n = static_cast<std::uint32_t>(entries_.size());
  out.write(reinterpret_cast<const char *>(&n), 4);
  for (const auto &kv : entries_) {
    std::uint16_t player_move;
    std::uint8_t opp_count = 0;
    std::uint8_t our_b = 0;
    if (ignore_opp_count_) {
      player_move = static_cast<std::uint16_t>(kv.first);
    } else {
      std::uint32_t k = kv.first;
      our_b = static_cast<std::uint8_t>(k % kOurBuckets);
      k /= kOurBuckets;
      opp_count = static_cast<std::uint8_t>(k % 17u);
      player_move = static_cast<std::uint16_t>(k / 17u);
    }
    out.write(reinterpret_cast<const char *>(&player_move), 2);
    out.write(reinterpret_cast<const char *>(&opp_count), 1);
    out.write(reinterpret_cast<const char *>(&our_b), 1);
    const Entry &e = kv.second;
    out.write(reinterpret_cast<const char *>(&e.pass_count), 4);
    out.write(reinterpret_cast<const char *>(&e.pass_trial), 4);
    out.write(reinterpret_cast<const char *>(e.main_count.data()), sizeof(float) * kRanks);
    out.write(reinterpret_cast<const char *>(e.main_trial.data()), sizeof(float) * kRanks);
    out.write(reinterpret_cast<const char *>(e.bomb_rank_count.data()), sizeof(float) * kRanks);
    out.write(reinterpret_cast<const char *>(e.bomb_rank_trial.data()), sizeof(float) * kRanks);
    out.write(reinterpret_cast<const char *>(e.bomb_aux_count.data()), sizeof(float) * kAuxRanks);
    out.write(reinterpret_cast<const char *>(e.bomb_aux_trial.data()), sizeof(float) * kAuxRanks);
    out.write(reinterpret_cast<const char *>(e.fh_aux_count.data()), sizeof(float) * kRanks);
    out.write(reinterpret_cast<const char *>(e.fh_aux_trial.data()), sizeof(float) * kRanks);
  }
}

MoveProbComponents MoveProbTable::query_components(
    int player_move, int opp_count, int our_hand_size_bucket,
    float kappa) const {
  // Start with prior from fallback (or default uniform).
  MoveProbComponents cv;
  if (fallback_) {
    cv = fallback_->query_components(player_move, opp_count,
                                       our_hand_size_bucket, kappa);
  }
  std::uint32_t key = make_key(player_move, opp_count, our_hand_size_bucket);
  auto it = entries_.find(key);
  if (it == entries_.end()) return cv;

  const Entry &e = it->second;
  cv.pass = (kappa * cv.pass + e.pass_count) / (kappa + e.pass_trial);
  for (int R = 0; R < kRanks; ++R) {
    cv.main_rank[R] =
        (kappa * cv.main_rank[R] + e.main_count[R]) /
        (kappa + e.main_trial[R]);
    cv.bomb_rank[R] =
        (kappa * cv.bomb_rank[R] + e.bomb_rank_count[R]) /
        (kappa + e.bomb_rank_trial[R]);
    cv.fh_aux[R] =
        (kappa * cv.fh_aux[R] + e.fh_aux_count[R]) /
        (kappa + e.fh_aux_trial[R]);
  }
  for (int A = 0; A < kAuxRanks; ++A) {
    cv.bomb_aux[A] =
        (kappa * cv.bomb_aux[A] + e.bomb_aux_count[A]) /
        (kappa + e.bomb_aux_trial[A]);
  }
  return cv;
}

void MoveProbTable::query(int player_move, int opp_count,
                           int our_hand_size_bucket,
                           const std::vector<int> &legal_moves,
                           std::vector<float> &out_probs) const {
  stats_.queries.fetch_add(1, std::memory_order_relaxed);
  out_probs.assign(legal_moves.size(), 0.0f);
  if (legal_moves.empty()) return;

  MoveProbComponents cv = query_components(player_move, opp_count,
                                              our_hand_size_bucket);
  double sum = 0.0;
  for (std::size_t i = 0; i < legal_moves.size(); ++i) {
    float w = move_prob_weight_of(legal_moves[i], player_move, cv);
    out_probs[i] = w;
    sum += w;
  }
  if (sum > 0.0) {
    float inv = static_cast<float>(1.0 / sum);
    for (auto &p : out_probs) p *= inv;
    stats_.main_hits.fetch_add(1, std::memory_order_relaxed);
  } else {
    // Degenerate fallback: uniform.
    float u = 1.0f / static_cast<float>(legal_moves.size());
    for (auto &p : out_probs) p = u;
    stats_.uniform.fetch_add(1, std::memory_order_relaxed);
  }
}

namespace {
// Factor a response y into the components it contributes to. Returns booleans
// for which components are touched and the relevant indices.
struct ResponseFactor {
  bool is_pass = false;
  int main_idx = -1;
  int bomb_rank_idx = -1;
  int bomb_aux_idx = -1;
  int fh_aux_idx = -1;
};
ResponseFactor factor_response(int y) {
  ResponseFactor r;
  if (y == kPASS) {
    r.is_pass = true;
    return r;
  }
  const Move &m = all_moves()[y];
  if (m.combination == Move::Combination::kBomb) {
    r.bomb_rank_idx = rank_idx_of_face(m.rank);
    r.bomb_aux_idx = bomb_aux_idx_of(m.auxiliary);
    return r;
  }
  // Same-type response.
  r.main_idx = rank_idx_of_face(m.rank);
  if (m.combination == Move::Combination::kFullHouse) {
    r.fh_aux_idx = rank_idx_of_face(m.auxiliary);
  }
  return r;
}
}  // namespace

void MoveProbTable::add_observation(int player_move, int opp_count,
                                     int our_hand_size_bucket,
                                     const std::vector<int> &feasible_set,
                                     int played_response, float weight) {
  std::uint32_t key = make_key(player_move, opp_count, our_hand_size_bucket);
  Entry &e = entries_[key];

  // Trials: increment each component-index touched by ANY feasible y at most
  // once per call. Set-track per dimension to avoid double counting when many
  // feasible responses share components.
  bool pass_seen = false;
  std::array<bool, kRanks> main_seen{}, bomb_rank_seen{}, fh_aux_seen{};
  std::array<bool, kAuxRanks> bomb_aux_seen{};
  for (int y : feasible_set) {
    ResponseFactor f = factor_response(y);
    if (f.is_pass) pass_seen = true;
    if (f.main_idx >= 0 && f.main_idx < kRanks) main_seen[f.main_idx] = true;
    if (f.bomb_rank_idx >= 0 && f.bomb_rank_idx < kRanks)
      bomb_rank_seen[f.bomb_rank_idx] = true;
    if (f.bomb_aux_idx >= 0 && f.bomb_aux_idx < kAuxRanks)
      bomb_aux_seen[f.bomb_aux_idx] = true;
    if (f.fh_aux_idx >= 0 && f.fh_aux_idx < kRanks)
      fh_aux_seen[f.fh_aux_idx] = true;
  }
  if (pass_seen) e.pass_trial += weight;
  for (int R = 0; R < kRanks; ++R) {
    if (main_seen[R]) e.main_trial[R] += weight;
    if (bomb_rank_seen[R]) e.bomb_rank_trial[R] += weight;
    if (fh_aux_seen[R]) e.fh_aux_trial[R] += weight;
  }
  for (int A = 0; A < kAuxRanks; ++A) {
    if (bomb_aux_seen[A]) e.bomb_aux_trial[A] += weight;
  }

  // Counts: only the played response contributes.
  if (played_response >= 0) {
    ResponseFactor f = factor_response(played_response);
    if (f.is_pass) e.pass_count += weight;
    if (f.main_idx >= 0 && f.main_idx < kRanks)
      e.main_count[f.main_idx] += weight;
    if (f.bomb_rank_idx >= 0 && f.bomb_rank_idx < kRanks)
      e.bomb_rank_count[f.bomb_rank_idx] += weight;
    if (f.bomb_aux_idx >= 0 && f.bomb_aux_idx < kAuxRanks)
      e.bomb_aux_count[f.bomb_aux_idx] += weight;
    if (f.fh_aux_idx >= 0 && f.fh_aux_idx < kRanks)
      e.fh_aux_count[f.fh_aux_idx] += weight;
  }
}

void MoveProbTable::decay(float alpha) {
  for (auto &kv : entries_) {
    Entry &e = kv.second;
    e.pass_count *= alpha;
    e.pass_trial *= alpha;
    for (int R = 0; R < kRanks; ++R) {
      e.main_count[R] *= alpha;
      e.main_trial[R] *= alpha;
      e.bomb_rank_count[R] *= alpha;
      e.bomb_rank_trial[R] *= alpha;
      e.fh_aux_count[R] *= alpha;
      e.fh_aux_trial[R] *= alpha;
    }
    for (int A = 0; A < kAuxRanks; ++A) {
      e.bomb_aux_count[A] *= alpha;
      e.bomb_aux_trial[A] *= alpha;
    }
  }
}

}  // namespace typed_search
