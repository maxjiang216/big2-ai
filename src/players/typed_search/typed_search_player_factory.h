#ifndef TYPED_SEARCH_TYPED_SEARCH_PLAYER_FACTORY_H
#define TYPED_SEARCH_TYPED_SEARCH_PLAYER_FACTORY_H

#include "player_factory.h"
#include "typed_search_player.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace typed_search {

// Factory for TypedSearchPlayer.
//
// Loads four table files from `dir` (default "data/typed_search"):
//   eval_main.bin
//   eval_fallback.bin
//   mp_main.bin
//   mp_fallback.bin
// Tables that don't exist are left empty (queries will fall through to
// uniform/baseline).
//
// All players returned by create_player() share the same loaded tables (via
// shared_ptr<const TypedSearchTables>). Each player gets a distinct RNG seed.
class TypedSearchPlayerFactory : public PlayerFactory {
public:
  explicit TypedSearchPlayerFactory(std::uint64_t base_seed = 0,
                                     const std::string &dir = "data/typed_search")
      : base_seed_(base_seed), dir_(dir),
        tables_(std::make_shared<TypedSearchTables>()) {
    auto t = std::const_pointer_cast<TypedSearchTables>(tables_);
    t->eval_extended.load(dir_ + "/eval_extended.bin");
    t->eval_main.load(dir_ + "/eval_main.bin");
    t->eval_fallback.load(dir_ + "/eval_fallback.bin");
    t->mp_main.load(dir_ + "/mp_main.bin");
    t->mp_fallback.load(dir_ + "/mp_fallback.bin");
  }

  std::unique_ptr<::Player> create_player() override {
    std::uint64_t seed = base_seed_ ^ (0xBF58476D1CE4E5B9ull * (++counter_));
    return std::make_unique<TypedSearchPlayer>(tables_, seed);
  }

private:
  std::uint64_t base_seed_;
  std::string dir_;
  std::shared_ptr<const TypedSearchTables> tables_;
  std::atomic<std::uint64_t> counter_{0};
};

}  // namespace typed_search

#endif
