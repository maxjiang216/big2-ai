// game_record.cpp
#include "game_record.h"
#include "game.h"
#include <arrow/api.h>
#include <sstream>

GameRecord::GameRecord() {}

void GameRecord::set_initial_state(const Game &game) {
    for(int p=0; p<2; ++p) {
        auto hand_vec = game.get_player_hand(p);
        for(int r=0; r<13; ++r) {
            _initial_hands[p][r] = hand_vec[r];
        }
    }
}

void GameRecord::add_move(const Move &move) {
    _moves.push_back(move);
}

std::string GameRecord::to_json() const {
    std::ostringstream oss;
    oss << "{\"initial_hands\": [";
    for(int p=0; p<2; ++p) {
        oss << "[";
        for(int r=0; r<13; ++r) {
            oss << _initial_hands[p][r];
            if(r<12) oss << ",";
        }
        oss << "]";
        if(p==0) oss << ",";
    }
    oss << "], \"moves\": [";
    for(size_t i=0; i<_moves.size(); ++i) {
        const auto &m = _moves[i];
        oss << "{"
            << "\"combination\":" << static_cast<int>(m.combination) << ","
            << "\"rank\":" << m.rank << ","
            << "\"auxiliary\":" << m.auxiliary
            << "}";
        if(i+1<_moves.size()) oss << ",";
    }
    oss << "], \"winner\":null}";
    return oss.str();
}

std::shared_ptr<arrow::Schema> GameRecord::arrow_schema() {
    auto int_type = arrow::int32();
    auto hand_type = arrow::list(int_type);
    auto f0 = arrow::field("initial_hand_p0", hand_type);
    auto f1 = arrow::field("initial_hand_p1", hand_type);

    std::vector<std::shared_ptr<arrow::Field>> mv_fields = {
        arrow::field("player",      arrow::int32()),
        arrow::field("combination", arrow::int32()),
        arrow::field("rank",        arrow::int32()),
        arrow::field("auxiliary",   arrow::int32())
    };
    auto mv_struct = arrow::struct_(mv_fields);
    auto fm = arrow::field("moves", arrow::list(mv_struct));

    return arrow::schema({f0,f1,fm});
}

arrow::Result<std::shared_ptr<arrow::Table>> GameRecord::ToTable(
    const std::vector<GameRecord>& records
) {
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    // Hand builders
    arrow::Int32Builder hand_val(pool);
    arrow::ListBuilder hand0_builder(pool, std::make_shared<arrow::Int32Builder>(pool));
    arrow::ListBuilder hand1_builder(pool, std::make_shared<arrow::Int32Builder>(pool));
    // Moves builder
    std::vector<std::unique_ptr<arrow::ArrayBuilder>> children;
    children.push_back(std::make_unique<arrow::Int32Builder>(pool)); // player
    children.push_back(std::make_unique<arrow::Int32Builder>(pool)); // comb
    children.push_back(std::make_unique<arrow::Int32Builder>(pool)); // rank
    children.push_back(std::make_unique<arrow::Int32Builder>(pool)); // aux
    auto mv_struct_builder = std::make_shared<arrow::StructBuilder>(
        arrow::struct_(mv_fields), pool, std::move(children)
    );
    arrow::ListBuilder moves_builder(pool, mv_struct_builder);

    for (const auto &rec : records) {
        // hand0
        ARROW_RETURN_NOT_OK(hand0_builder.Append());
        auto hv0 = static_cast<arrow::Int32Builder*>(hand0_builder.value_builder());
        for(int i=0;i<13;++i) ARROW_RETURN_NOT_OK(hv0->Append(rec._initial_hands[0][i]));
        // hand1
        ARROW_RETURN_NOT_OK(hand1_builder.Append());
        auto hv1 = static_cast<arrow::Int32Builder*>(hand1_builder.value_builder());
        for(int i=0;i<13;++i) ARROW_RETURN_NOT_OK(hv1->Append(rec._initial_hands[1][i]));
        // moves
        ARROW_RETURN_NOT_OK(moves_builder.Append());
        auto sb = static_cast<arrow::StructBuilder*>(moves_builder.value_builder());
        auto pb = static_cast<arrow::Int32Builder*>(sb->child_builder(0));
        auto cb = static_cast<arrow::Int32Builder*>(sb->child_builder(1));
        auto rb = static_cast<arrow::Int32Builder*>(sb->child_builder(2));
        auto ab = static_cast<arrow::Int32Builder*>(sb->child_builder(3));
        for (const auto &m : rec._moves) {
            ARROW_RETURN_NOT_OK(sb->Append());
            ARROW_RETURN_NOT_OK(pb->Append(m.player));
            ARROW_RETURN_NOT_OK(cb->Append(static_cast<int>(m.combination)));
            ARROW_RETURN_NOT_OK(rb->Append(m.rank));
            ARROW_RETURN_NOT_OK(ab->Append(m.auxiliary));
        }
    }
    std::shared_ptr<arrow::Array> a0, a1, am;
    ARROW_RETURN_NOT_OK(hand0_builder.Finish(&a0));
    ARROW_RETURN_NOT_OK(hand1_builder.Finish(&a1));
    ARROW_RETURN_NOT_OK(moves_builder.Finish(&am));

    return arrow::Table::Make(arrow_schema(), {a0,a1,am});
}
