#include <iostream>

void run_game_tests();
void run_move_tests();
void run_partial_game_tests();
void run_greedy_player_tests();
void run_random_game_tests();
void run_tablebase_tests();

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

  std::cout << "All tests passed.\n";
  return 0;
}
