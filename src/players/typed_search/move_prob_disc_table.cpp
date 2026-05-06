#include "move_prob_disc_table.h"

#include "move.h"
#include "util.h"

#include <algorithm>
#include <cstdint>
#include <fstream>

namespace typed_search {

namespace {
// Bumped to v3: factored per-component representation.
constexpr char kMagic[4] = {'M', 'P', 'D', '3'};

inline int rank_idx_of_face(int face_rank) {
  return (face_rank == 2 || face_rank == 15) ? 12 : (face_rank - 3);
}

inline int bomb_aux_idx_of(int aux_face) {
  if (aux_face == 0) return 0;
  return rank_idx_of_face(aux_face) + 1;
}

struct ResponseFactor {
  bool is_pass = false;
  int main_idx = -1;
  int bomb_rank_idx = -1;
  int bomb_aux_idx = -1;
  int fh_aux_idx = -1;
};
ResponseFactor factor_response(int y) {
  ResponseFactor r;
  if (y == kPASS) { r.is_pass = true; return r; }
  const Move &m = all_moves()[y];
  if (m.combination == Move::Combination::kBomb) {
    r.bomb_rank_idx = rank_idx_of_face(m.rank);
    r.bomb_aux_idx = bomb_aux_idx_of(m.auxiliary);
    return r;
  }
  r.main_idx = rank_idx_of_face(m.rank);
  if (m.combination == Move::Combination::kFullHouse) {
    r.fh_aux_idx = rank_idx_of_face(m.auxiliary);
  }
  return r;
}
}  // namespace

bool MoveProbDiscardTable::is_eligible(int move_id) {
  // singles 3-K = move 1..11; doubles 3-K = 14..24; triples 3-Q = 26..35.
  return (move_id >= 1 && move_id <= 11) ||
          (move_id >= 14 && move_id <= 24) ||
          (move_id >= 26 && move_id <= 35);
}

std::uint16_t MoveProbDiscardTable::compute_bitmap(
    const std::array<int, 13> &hand_after,
    const std::array<int, 13> &discard_after, int opp_count) {
  std::uint16_t bits = 0;
  for (int r_idx = 1; r_idx <= 11; ++r_idx) {
    int max_in_deck = max_cards_in_deck_for_rank(r_idx);
    int opp_max = max_in_deck - hand_after[r_idx] - discard_after[r_idx];
    if (opp_max > opp_count) opp_max = opp_count;
    if (opp_max >= 3) bits |= static_cast<std::uint16_t>(1u << (r_idx - 1));
  }
  return bits;
}

std::uint16_t MoveProbDiscardTable::mask_relevant(int move_id,
                                                    std::uint16_t bitmap) {
  Move m(move_id);
  int rank_idx = rank_idx_of_face(m.rank);
  if (rank_idx >= kBitmapWidth) return 0;
  std::uint16_t mask =
      static_cast<std::uint16_t>(~((1u << rank_idx) - 1u)) & 0x7FFu;
  // Triples: drop the A-bit (bit 10). Triple A is a bomb response, not a
  // triple, so the A bit doesn't condition triple responses.
  if (m.combination == Move::Combination::kTriple) {
    mask = static_cast<std::uint16_t>(mask & 0x3FFu);  // clear bit 10
  }
  return static_cast<std::uint16_t>(bitmap & mask);
}

void MoveProbDiscardTable::load(const std::string &path) {
  entries_.clear();
  std::ifstream in(path, std::ios::binary);
  if (!in) return;
  char magic[4];
  in.read(magic, 4);
  if (in.gcount() != 4 || magic[0] != kMagic[0] || magic[1] != kMagic[1] ||
      magic[2] != kMagic[2] || magic[3] != kMagic[3]) {
    return;
  }
  std::uint32_t n = 0;
  in.read(reinterpret_cast<char *>(&n), 4);
  if (!in) return;
  entries_.reserve(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint32_t key = 0;
    in.read(reinterpret_cast<char *>(&key), 4);
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
    entries_[key] = std::move(e);
  }
}

void MoveProbDiscardTable::save(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  out.write(kMagic, 4);
  std::uint32_t n = static_cast<std::uint32_t>(entries_.size());
  out.write(reinterpret_cast<const char *>(&n), 4);
  for (const auto &kv : entries_) {
    std::uint32_t key = kv.first;
    out.write(reinterpret_cast<const char *>(&key), 4);
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

void MoveProbDiscardTable::query(int move_id, int opp_count,
                                   int our_hand_size_bucket,
                                   std::uint16_t bitmap_full,
                                   const std::vector<int> &legal_moves,
                                   const MoveProbTable &mp_main,
                                   std::vector<float> &out_probs) const {
  // Get mp_main's components as the prior.
  MoveProbComponents cv = mp_main.query_components(move_id, opp_count,
                                                      our_hand_size_bucket);

  // If eligible and we have data in this cell, shrink each component
  // toward mp_main's value with disc data.
  if (is_eligible(move_id)) {
    std::uint16_t rel = mask_relevant(move_id, bitmap_full);
    std::uint32_t key = make_key(move_id, opp_bucket(opp_count),
                                    our_hand_size_bucket, rel);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      constexpr float kKappa = 20.0f;
      const Entry &e = it->second;
      cv.pass = (kKappa * cv.pass + e.pass_count) / (kKappa + e.pass_trial);
      for (int R = 0; R < kRanks; ++R) {
        cv.main_rank[R] =
            (kKappa * cv.main_rank[R] + e.main_count[R]) /
            (kKappa + e.main_trial[R]);
        cv.bomb_rank[R] =
            (kKappa * cv.bomb_rank[R] + e.bomb_rank_count[R]) /
            (kKappa + e.bomb_rank_trial[R]);
        cv.fh_aux[R] = (kKappa * cv.fh_aux[R] + e.fh_aux_count[R]) /
                        (kKappa + e.fh_aux_trial[R]);
      }
      for (int A = 0; A < kAuxRanks; ++A) {
        cv.bomb_aux[A] = (kKappa * cv.bomb_aux[A] + e.bomb_aux_count[A]) /
                          (kKappa + e.bomb_aux_trial[A]);
      }
    }
  }

  // Compute per-y weights from the (possibly disc-updated) components,
  // normalize.
  out_probs.assign(legal_moves.size(), 0.0f);
  if (legal_moves.empty()) return;
  double sum = 0.0;
  for (std::size_t i = 0; i < legal_moves.size(); ++i) {
    float w = move_prob_weight_of(legal_moves[i], move_id, cv);
    out_probs[i] = w;
    sum += w;
  }
  if (sum > 0.0) {
    float inv = static_cast<float>(1.0 / sum);
    for (auto &p : out_probs) p *= inv;
  } else {
    float u = 1.0f / static_cast<float>(legal_moves.size());
    for (auto &p : out_probs) p = u;
  }
}

void MoveProbDiscardTable::add_observation(int move_id, int opp_count,
                                             int our_hand_size_bucket,
                                             std::uint16_t bitmap_full,
                                             const std::vector<int> &feasible_set,
                                             int played_response,
                                             float weight) {
  if (!is_eligible(move_id)) return;
  std::uint16_t rel = mask_relevant(move_id, bitmap_full);
  std::uint32_t key = make_key(move_id, opp_bucket(opp_count),
                                 our_hand_size_bucket, rel);
  Entry &e = entries_[key];

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

void MoveProbDiscardTable::decay(float alpha) {
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
