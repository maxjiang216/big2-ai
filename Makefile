# Build and utility targets for the big2-ai Rust workspace.
# All game engine, players, simulation, and datagen code is now in crates/.

REPO_ROOT := $(shell pwd)

.PHONY: all build release test tablebase_opp1_gen standards_init standards_configs help clean

all: help

help:
	@echo "Targets:"
	@echo "  make build             - cargo build (debug)"
	@echo "  make release           - cargo build --release (generates eval_match, generate_data)"
	@echo "  make test              - cargo test --all"
	@echo "  make tablebase_opp1_gen - build the one-card tablebase generator (standalone C++)"
	@echo "  make standards_init    - git submodule: fetch projects/standard-linter"
	@echo "  make standards_configs - copy configs from projects/standard-linter → .code-standards/"
	@echo "  make clean             - remove target/"
	@echo ""
	@echo "Binaries after 'make release':"
	@echo "  target/release/eval_match      - head-to-head evaluation (Wilson 95% CI)"
	@echo "  target/release/generate_data   - self-play + Parquet export"

build:
	cargo build

release:
	cargo build --release
	@echo "✓ target/release/eval_match"
	@echo "✓ target/release/generate_data"

test:
	cargo test --all

# ============================================================================
# bin/tablebase_opp1_gen — standalone C++ (no Rust equivalent; precomputes
# the opp-1-card tablebase .bin file that the Rust tablebase loader reads).
# ============================================================================

BIN_DIR = bin

$(BIN_DIR)/tablebase_opp1_gen: scripts/tablebase_opp1_gen.cpp | $(BIN_DIR)
	g++ -O2 -std=c++17 -o $@ $<
	@echo "✓ $(BIN_DIR)/tablebase_opp1_gen"

$(BIN_DIR):
	@mkdir -p $@

tablebase_opp1_gen: $(BIN_DIR)/tablebase_opp1_gen

# ============================================================================
# Shared code standards (vendored git submodule: github.com/maxjiang216/standard-linter)
# ============================================================================

standards_init:
	git submodule update --init projects/standard-linter

standards_configs: standards_init
	DEST="$(REPO_ROOT)/.code-standards" bash projects/standard-linter/scripts/install-local-standards.sh

# ============================================================================

clean:
	cargo clean
	rm -rf $(BIN_DIR)
