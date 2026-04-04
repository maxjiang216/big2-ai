#include "samples_md.h"

#include "game.h"
#include "move.h"
#include "util.h"

#include <algorithm>
#include <iostream>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string hand_rank_string(const Game &g, int player) {
  std::string s;
  auto h = g.player_hand(player);
  for (int i = 0; i < 13; ++i) {
    for (int c = 0; c < h[i]; ++c)
      s += rankToChar(i + 3);
  }
  return s;
}

static int start_legal_moves_count(const GameRecord &rec) {
  const auto &turns = rec.turns();
  if (turns.empty())
    return -1;
  return static_cast<int>(turns[0].legal_moves.size());
}

bool write_samples_md(const std::vector<std::pair<int, GameRecord>> &indexed,
                      const std::string &path) {
  if (indexed.empty())
    return true;
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Error: cannot open --samples-md path '" << path << "'\n";
    return false;
  }

  int best_long = -1, best_long_len = -1;
  int best_short = -1, best_short_len = INT_MAX;
  int best_start_g = -1, best_start_n = -1;
  int worst_start_g = -1, worst_start_n = INT_MAX;

  for (const auto &pr : indexed) {
    int gi = pr.first;
    const auto &rec = pr.second;
    int L = static_cast<int>(rec.turns().size());
    int sl = start_legal_moves_count(rec);
    if (L > best_long_len) {
      best_long_len = L;
      best_long = gi;
    }
    if (L < best_short_len) {
      best_short_len = L;
      best_short = gi;
    }
    if (sl >= 0) {
      if (sl > best_start_n) {
        best_start_n = sl;
        best_start_g = gi;
      }
      if (sl < worst_start_n) {
        worst_start_n = sl;
        worst_start_g = gi;
      }
    }
  }

  out << "# Sample games (anomalies)\n\n";
  out << "From this run: **longest** / **shortest** game by turn count; **most** / **fewest** "
         "legal moves for the opening player before the first card "
         "(same as feature `start_legal_moves`).\n\n";
  out << "`game_index` is the global game number for this run (0 … N−1).\n\n";

  auto labels_for = [&](int gi) {
    std::vector<std::string> labs;
    if (gi == best_long)
      labs.push_back("longest game");
    if (gi == best_short)
      labs.push_back("shortest game");
    if (gi == best_start_g)
      labs.push_back("most opening legal moves");
    if (gi == worst_start_g)
      labs.push_back("fewest opening legal moves");
    return labs;
  };

  std::vector<int> emit_order;
  for (int cand : {best_long, best_short, best_start_g, worst_start_g}) {
    if (cand < 0)
      continue;
    if (std::find(emit_order.begin(), emit_order.end(), cand) != emit_order.end())
      continue;
    emit_order.push_back(cand);
  }

  for (int gi : emit_order) {
    const GameRecord *rec_ptr = nullptr;
    for (const auto &pr : indexed) {
      if (pr.first == gi) {
        rec_ptr = &pr.second;
        break;
      }
    }
    if (!rec_ptr)
      continue;
    const GameRecord &rec = *rec_ptr;
    auto labs = labels_for(gi);
    out << "## Game `" << gi << "`";
    if (!labs.empty()) {
      out << " — *";
      for (size_t i = 0; i < labs.size(); ++i) {
        if (i)
          out << ", ";
        out << labs[i];
      }
      out << "*";
    }
    out << "\n\n";

    int winner = rec.game().get_winner();
    int n_turns = static_cast<int>(rec.turns().size());
    int slm = start_legal_moves_count(rec);
    out << "- **Turns:** " << n_turns << "\n";
    out << "- **Winner:** P" << winner << "\n";
    if (slm >= 0)
      out << "- **Opening legal moves (start_legal_moves):** " << slm << "\n";
    out << "\n";

    if (rec.turns().empty()) {
      out << "*(No turns recorded.)*\n\n";
      continue;
    }

    const Game &g0 = rec.turns()[0].game;
    out << "### Starting hands (before move 0)\n\n";
    out << "| Player | Cards (rank symbols; 3 lowest … 2 highest) |\n";
    out << "|--------|--------------------------------------------|\n";
    out << "| P0 | `" << hand_rank_string(g0, 0) << "` |\n";
    out << "| P1 | `" << hand_rank_string(g0, 1) << "` |\n\n";

    out << "### Turn-by-turn history\n\n";
    out << "| Turn | Player to move | Move | `move_id` |\n";
    out << "|------|----------------|------|------------|\n";
    for (int ti = 0; ti < n_turns; ++ti) {
      const auto &turn = rec.turns()[ti];
      std::ostringstream mv;
      mv << turn.move;
      out << "| " << ti << " | P" << turn.current_player << " | `" << mv.str() << "` | "
          << encodeMove(turn.move) << " |\n";
    }
    out << "\n";
  }

  out.close();
  return true;
}
