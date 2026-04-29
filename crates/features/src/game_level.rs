use big2_core::{GameRecord, tablebase_first_hit_stats};

pub trait GameLevelFeature: Send + Sync {
    fn name(&self) -> &'static str;
    fn extract(&self, record: &GameRecord) -> i32;
}

// ---------------------------------------------------------------------------

pub struct OutcomeFeature;
impl GameLevelFeature for OutcomeFeature {
    fn name(&self) -> &'static str { "outcome" }
    fn extract(&self, record: &GameRecord) -> i32 { record.winner() as i32 }
}

pub struct LengthFeature;
impl GameLevelFeature for LengthFeature {
    fn name(&self) -> &'static str { "length" }
    fn extract(&self, record: &GameRecord) -> i32 { record.turns.len() as i32 }
}

pub struct StartLegalMovesFeature;
impl GameLevelFeature for StartLegalMovesFeature {
    fn name(&self) -> &'static str { "start_legal_moves" }
    fn extract(&self, record: &GameRecord) -> i32 {
        record.turns.first().map(|t| t.legal_moves.len() as i32).unwrap_or(-1)
    }
}

pub struct TablebaseHitsFeature;
impl GameLevelFeature for TablebaseHitsFeature {
    fn name(&self) -> &'static str { "tb_hits" }
    fn extract(&self, record: &GameRecord) -> i32 {
        let s = tablebase_first_hit_stats(record);
        (s.case1 + s.case2) as i32
    }
}

pub struct TbCase1Feature;
impl GameLevelFeature for TbCase1Feature {
    fn name(&self) -> &'static str { "tb_case1" }
    fn extract(&self, record: &GameRecord) -> i32 {
        tablebase_first_hit_stats(record).case1 as i32
    }
}

pub struct TbCase2Feature;
impl GameLevelFeature for TbCase2Feature {
    fn name(&self) -> &'static str { "tb_case2" }
    fn extract(&self, record: &GameRecord) -> i32 {
        tablebase_first_hit_stats(record).case2 as i32
    }
}

pub struct TbForcedSeqLenFeature;
impl GameLevelFeature for TbForcedSeqLenFeature {
    fn name(&self) -> &'static str { "tb_forced_seq_len" }
    fn extract(&self, record: &GameRecord) -> i32 {
        tablebase_first_hit_stats(record).case1_seq_max as i32
    }
}

pub struct TbForcedSeqLenSumFeature;
impl GameLevelFeature for TbForcedSeqLenSumFeature {
    fn name(&self) -> &'static str { "tb_forced_seq_len_sum" }
    fn extract(&self, record: &GameRecord) -> i32 {
        tablebase_first_hit_stats(record).case1_seq_sum as i32
    }
}

pub struct TbOpp1TableStraightFeature;
impl GameLevelFeature for TbOpp1TableStraightFeature {
    fn name(&self) -> &'static str { "tb_opp1_table_straight" }
    fn extract(&self, record: &GameRecord) -> i32 {
        tablebase_first_hit_stats(record).case2_table_straight as i32
    }
}
