// Cross-check az_search C++ inference against the Python forward pass.
//
//   make az_nn_check
//   ./bin/az_nn_check models/az_player.pt models/az_opp.pt
//   uv run python -m nn.az_nn_check models/az_player.pt models/az_opp.pt
//
// Both print value + the first few logits + argmax for a fixed set of positions
// defined identically on both sides; the numbers must agree to fp tolerance.

#include "az_search/nn_eval.h"

#include <cstdio>
#include <vector>

using namespace az_search;

// Fixed test positions (mirrored in nn/az_nn_check.py).
static std::vector<PlayerFeatures> player_positions() {
  return {
      {{2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1}, // hand
       {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0}, // opp_max
       {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // trick (single 5)
       8,
       6},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // empty hand
       {4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // lead
       3,
       0},
      {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, // one of each
       {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0},
       {0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0}, // a pair of 10s
       11,
       13},
  };
}

static std::vector<OppFeatures> opp_positions() {
  return {
      {{2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1}, // observer's hand
       {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0}, // opp_max (thermo)
       {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // trick (single 5)
       8,
       6},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // empty hand
       {4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // lead
       3,
       0},
      {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, // one of each
       {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0},
       {0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0}, // a pair of 10s
       11,
       13},
  };
}

template <int N>
static int argmax(const std::array<float, N> &a) {
  int best = 0;
  for (int i = 1; i < N; ++i)
    if (a[i] > a[best]) best = i;
  return best;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s player.pt opp.pt\n", argv[0]);
    return 2;
  }
  NNEvaluator nn(argv[1], argv[2], torch::Device(torch::kCPU));

  auto pe = nn.eval_players(player_positions());
  std::printf("PLAYER\n");
  for (size_t i = 0; i < pe.size(); ++i)
    std::printf("  pos%zu value=%.6f logits[0..4]=%.6f %.6f %.6f %.6f %.6f argmax=%d\n",
                i, pe[i].value, pe[i].logits[0], pe[i].logits[1], pe[i].logits[2],
                pe[i].logits[3], pe[i].logits[4],
                argmax<AZ_PLAYER_HEAD_DIM>(pe[i].logits));

  auto oe = nn.eval_opps(opp_positions());
  std::printf("OPP\n");
  for (size_t i = 0; i < oe.size(); ++i)
    std::printf("  pos%zu move_value[0..4]=%.6f %.6f %.6f %.6f %.6f "
                "logits[0..4]=%.6f %.6f %.6f %.6f %.6f argmax=%d\n",
                i, oe[i].move_value[0], oe[i].move_value[1], oe[i].move_value[2],
                oe[i].move_value[3], oe[i].move_value[4], oe[i].logits[0],
                oe[i].logits[1], oe[i].logits[2], oe[i].logits[3], oe[i].logits[4],
                argmax<AZ_OPP_HEAD_DIM>(oe[i].logits));
  return 0;
}
