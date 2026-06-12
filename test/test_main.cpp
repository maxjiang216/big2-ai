#include <iostream>

void run_game_tests();
void run_move_tests();
void run_partial_game_tests();
void run_greedy_player_tests();
void run_random_game_tests();
void run_tablebase_tests();
void run_legal_moves_tests();
void run_typed_search_tests();
void run_az_search_tests();
void run_az_pi_tests();
void run_series_tests();

int main() {
  std::cout << "[ move        ] ";
  run_move_tests();
  std::cout << "PASS\n";

  std::cout << "[ game        ] ";
  run_game_tests();
  std::cout << "PASS\n";

  std::cout << "[ partial     ] ";
  run_partial_game_tests();
  std::cout << "PASS\n";

  std::cout << "[ greedy      ] ";
  run_greedy_player_tests();
  std::cout << "PASS\n";

  std::cout << "[ random_games] ";
  run_random_game_tests();
  std::cout << "PASS\n";

  std::cout << "[ tablebase   ] ";
  run_tablebase_tests();
  std::cout << "PASS\n";

  std::cout << "[ legal_moves ] ";
  run_legal_moves_tests();
  std::cout << "PASS\n";

  std::cout << "[ typed_search] ";
  run_typed_search_tests();
  std::cout << "PASS\n";

  std::cout << "[ az_search   ] ";
  run_az_search_tests();
  std::cout << "PASS\n";

  std::cout << "[ az_pi       ] ";
  run_az_pi_tests();
  std::cout << "PASS\n";

  std::cout << "[ series      ] ";
  run_series_tests();
  std::cout << "PASS\n";

  std::cout << "All tests passed.\n";
  return 0;
}
