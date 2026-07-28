//! Counted, pruned time-range scan.
//!
//! Three layers, cheapest first — the point is the bytes never read:
//!   1. log-level file pruning (zone maps in the commit entries; zero I/O)
//!   2. Parquet row-group pruning (footer statistics; footer bytes only)
//!   3. projected reads of surviving row groups, row-filtered to [t0, t1]
//! Bounds are inclusive and `ts` always rides along, mirroring the Python
//! engine so results are comparable row for row.

use anyhow::{Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::SchemaRef;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ProjectionMask;
use parquet::file::statistics::Statistics;
use rayon::prelude::*;

use crate::count::{ByteCounter, CountingFile};
use crate::store::Store;

#[derive(Debug, Default, serde::Serialize)]
pub struct ScanStats {
    pub rows: u64,
    pub bytes_read: u64,
    pub bytes_total: u64,
    pub files_total: usize,
    pub files_scanned: usize,
    pub row_groups_total: usize,
    pub row_groups_scanned: usize,
}

impl ScanStats {
    pub fn elided_pct(&self) -> f64 {
        if self.bytes_total == 0 {
            return 0.0;
        }
        100.0 * (self.bytes_total.saturating_sub(self.bytes_read)) as f64
            / self.bytes_total as f64
    }
}

pub struct ScanResult {
    pub batches: Vec<RecordBatch>,
    pub schema: Option<SchemaRef>,
    pub stats: ScanStats,
}

pub fn scan(
    store: &Store,
    table: &str,
    t0: Option<i64>,
    t1: Option<i64>,
    columns: Option<&[String]>,
    version: Option<u64>,
) -> Result<ScanResult> {
    let log = store.log(table);
    let st = log.read_state(version)?;
    let mut stats = ScanStats {
        files_total: st.files.len(),
        bytes_total: st.bytes(),
        ..Default::default()
    };
    let counter = ByteCounter::new();
    let mut batches = Vec::new();
    let mut schema: Option<SchemaRef> = None;

    for f in &st.files {
        // layer 1: file-level zone maps from the log — no I/O at all
        if t0.is_some_and(|t| f.max_ts < t) || t1.is_some_and(|t| f.min_ts > t) {
            continue;
        }
        stats.files_scanned += 1;
        let path = log.dir.join(&f.path);
        let cf = CountingFile::open(&path, counter.clone())
            .with_context(|| format!("open {path:?}"))?;
        // footer read once (and counted once); per-row-group readers below
        // share this metadata instead of re-reading it
        let meta = ArrowReaderMetadata::load(&cf, ArrowReaderOptions::new())?;
        let md = meta.metadata().clone();
        let descr = md.file_metadata().schema_descr();

        // ts leaf index (top-level i64 column named "ts")
        let ts_leaf = (0..descr.num_columns())
            .find(|&i| descr.column(i).path().parts()[0] == "ts");

        // layer 2: row-group pruning on ts footer statistics
        let mut keep = Vec::new();
        stats.row_groups_total += md.num_row_groups();
        for rg in 0..md.num_row_groups() {
            let overlap = match ts_leaf {
                Some(ti) => match md.row_group(rg).column(ti).statistics() {
                    Some(Statistics::Int64(s)) => {
                        !(t0.is_some_and(|t| s.max_opt().is_some_and(|&m| m < t))
                            || t1.is_some_and(|t| s.min_opt().is_some_and(|&m| m > t)))
                    }
                    _ => true, // no stats: cannot prune, must read
                },
                None => true,
            };
            if overlap {
                keep.push(rg);
            }
        }
        stats.row_groups_scanned += keep.len();
        if keep.is_empty() {
            continue;
        }

        // projection: requested roots plus ts (the sort/alignment axis)
        let mask = match columns {
            Some(cols) => {
                let leaves: Vec<usize> = (0..descr.num_columns())
                    .filter(|&i| {
                        let root = descr.column(i).path().parts()[0].to_string();
                        root == "ts" || cols.iter().any(|c| *c == root)
                    })
                    .collect();
                ProjectionMask::leaves(descr, leaves)
            }
            None => ProjectionMask::all(),
        };

        // decode surviving row groups in parallel; each task gets its own
        // reader over the same counted file handle and shared footer
        let file_batches: Vec<Vec<RecordBatch>> = keep
            .par_iter()
            .map(|&rg| -> Result<Vec<RecordBatch>> {
                let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(
                    cf.clone(),
                    meta.clone(),
                )
                .with_row_groups(vec![rg])
                .with_projection(mask.clone())
                .with_batch_size(65_536)
                .build()?;
                let mut out = Vec::new();
                for batch in reader {
                    let batch = filter_ts(batch?, t0, t1)?;
                    if batch.num_rows() > 0 {
                        out.push(batch);
                    }
                }
                Ok(out)
            })
            .collect::<Result<_>>()?;
        for batch in file_batches.into_iter().flatten() {
            schema.get_or_insert_with(|| batch.schema());
            stats.rows += batch.num_rows() as u64;
            batches.push(batch);
        }
    }

    // files may interleave in time across streams: merge-sort by ts, but only
    // when more than one file actually contributed (single-file output is
    // already ts-sorted by the writer contract)
    if stats.files_scanned > 1 && batches.len() > 1 {
        batches = sort_by_ts(batches)?;
    }
    stats.bytes_read = counter.get();
    Ok(ScanResult { batches, schema, stats })
}

fn ts_column(batch: &RecordBatch) -> Result<&Int64Array> {
    let idx = batch
        .schema()
        .index_of("ts")
        .context("table has no ts column")?;
    Ok(batch.column(idx).as_primitive::<Int64Type>())
}

fn filter_ts(batch: RecordBatch, t0: Option<i64>, t1: Option<i64>) -> Result<RecordBatch> {
    if t0.is_none() && t1.is_none() {
        return Ok(batch);
    }
    let ts = ts_column(&batch)?;
    let mut mask: Option<arrow_array::BooleanArray> = None;
    if let Some(t) = t0 {
        mask = Some(arrow_ord::cmp::gt_eq(ts, &Int64Array::new_scalar(t))?);
    }
    if let Some(t) = t1 {
        let le = arrow_ord::cmp::lt_eq(ts, &Int64Array::new_scalar(t))?;
        mask = Some(match mask {
            Some(ge) => arrow_arith::boolean::and(&ge, &le)?,
            None => le,
        });
    }
    Ok(arrow_select::filter::filter_record_batch(&batch, &mask.unwrap())?)
}

fn sort_by_ts(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
    let schema = batches[0].schema();
    let all = arrow_select::concat::concat_batches(&schema, &batches)?;
    let ts = ts_column(&all)?;
    let idx = arrow_ord::sort::sort_to_indices(ts, None, None)?;
    let cols = all
        .columns()
        .iter()
        .map(|c| Ok(arrow_select::take::take(c, &idx, None)?))
        .collect::<Result<Vec<_>>>()?;
    Ok(vec![RecordBatch::try_new(schema, cols)?])
}
