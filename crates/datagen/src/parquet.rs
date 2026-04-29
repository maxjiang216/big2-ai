use std::fs::File;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use big2_core::GameRecord;
use big2_features::game_level::GameLevelFeature;
use big2_features::turn_level::TurnLevelFeature;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

/// Export a batch of indexed game records to two Parquet files:
///   `game_path` — one row per game: game_index + game-level features
///   `turn_path` — one row per (game, turn, perspective): turn-level features
pub fn export_parquet(
    indexed: &[(u32, GameRecord)],
    game_path: &str,
    turn_path: &str,
    game_features: &[Box<dyn GameLevelFeature>],
    turn_features: &[Box<dyn TurnLevelFeature>],
) -> Result<(), String> {
    export_game_table(indexed, game_path, game_features)?;
    export_turn_table(indexed, turn_path, turn_features)?;
    Ok(())
}

fn export_game_table(
    indexed: &[(u32, GameRecord)],
    path: &str,
    features: &[Box<dyn GameLevelFeature>],
) -> Result<(), String> {
    let n = indexed.len();

    let mut fields = vec![Field::new("game_index", DataType::Int32, false)];
    for f in features {
        fields.push(Field::new(f.name(), DataType::Int32, false));
    }
    let schema = Arc::new(Schema::new(fields));

    let game_idx: Vec<i32> = indexed.iter().map(|(i, _)| *i as i32).collect();
    let mut columns: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(game_idx))];

    for f in features {
        let vals: Vec<i32> = indexed.iter().map(|(_, r)| f.extract(r)).collect();
        columns.push(Arc::new(Int32Array::from(vals)));
    }

    let batch = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| format!("RecordBatch error: {}", e))?;

    let file = File::create(path).map_err(|e| format!("Cannot create {}: {}", path, e))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| format!("ArrowWriter error: {}", e))?;
    writer.write(&batch).map_err(|e| format!("Write error: {}", e))?;
    writer.close().map_err(|e| format!("Close error: {}", e))?;

    eprintln!("Exported {} game rows to {}", n, path);
    Ok(())
}

fn export_turn_table(
    indexed: &[(u32, GameRecord)],
    path: &str,
    features: &[Box<dyn TurnLevelFeature>],
) -> Result<(), String> {
    let mut game_idx_col: Vec<i32> = Vec::new();
    let mut turn_idx_col: Vec<i32> = Vec::new();
    let mut perspective_col: Vec<i32> = Vec::new();

    for (gi, record) in indexed {
        for (ti, _turn) in record.turns.iter().enumerate() {
            for p in 0..2i32 {
                game_idx_col.push(*gi as i32);
                turn_idx_col.push(ti as i32);
                perspective_col.push(p);
            }
        }
    }
    let total = game_idx_col.len();

    // Collect feature columns
    let mut feat_cols: Vec<Vec<i32>> = features.iter().map(|_| Vec::with_capacity(total)).collect();
    for (_, record) in indexed {
        for (fi, f) in features.iter().enumerate() {
            let vals = f.extract_all(record);
            feat_cols[fi].extend_from_slice(&vals);
        }
    }

    // Verify lengths
    for (fi, col) in feat_cols.iter().enumerate() {
        if col.len() != total {
            return Err(format!(
                "Feature {} has {} values, expected {}",
                features[fi].name(), col.len(), total
            ));
        }
    }

    let mut fields = vec![
        Field::new("game_index",  DataType::Int32, false),
        Field::new("turn_idx",    DataType::Int32, false),
        Field::new("perspective", DataType::Int32, false),
    ];
    for f in features {
        fields.push(Field::new(f.name(), DataType::Int32, false));
    }
    let schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(game_idx_col)),
        Arc::new(Int32Array::from(turn_idx_col)),
        Arc::new(Int32Array::from(perspective_col)),
    ];
    for col in feat_cols {
        columns.push(Arc::new(Int32Array::from(col)));
    }

    let batch = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| format!("RecordBatch error: {}", e))?;

    let file = File::create(path).map_err(|e| format!("Cannot create {}: {}", path, e))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| format!("ArrowWriter error: {}", e))?;
    writer.write(&batch).map_err(|e| format!("Write error: {}", e))?;
    writer.close().map_err(|e| format!("Close error: {}", e))?;

    eprintln!("Exported {} turn rows ({} games) to {}", total, indexed.len(), path);
    Ok(())
}
