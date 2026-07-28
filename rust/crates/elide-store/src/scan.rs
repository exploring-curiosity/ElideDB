//! Counted, pruned scan: time range plus arbitrary column predicates.
//!
//! Four layers, cheapest first — the point is the bytes never read:
//!   1. log-level file pruning (zone maps in the commit entries; zero I/O)
//!   2. Parquet row-group pruning (footer statistics; footer bytes only)
//!   3. Parquet PAGE pruning (column index) — a surviving row group
//!      still reads only the pages whose min/max can hold a match
//!   4. exact Arrow filtering of what actually got decoded
//! Bounds are inclusive and `ts` always rides along, mirroring the Python
//! engine so results are comparable row for row.

use anyhow::{Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::SchemaRef;
use parquet::arrow::arrow_reader::{
    ArrowPredicateFn, ArrowReaderMetadata, ArrowReaderOptions,
    ParquetRecordBatchReaderBuilder, RowFilter, RowSelection, RowSelector,
};
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::PageIndexPolicy;
use parquet::file::statistics::Statistics;
use rayon::prelude::*;

use crate::count::{ByteCounter, CountingFile};
use crate::predicate::{self, Predicate};
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
    pub pages_total: usize,
    pub pages_scanned: usize,
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
    scan_where(store, table, t0, t1, columns, version, &[])
}

#[allow(clippy::too_many_arguments)]
pub fn scan_where(
    store: &Store,
    table: &str,
    t0: Option<i64>,
    t1: Option<i64>,
    columns: Option<&[String]>,
    version: Option<u64>,
    preds: &[Predicate],
) -> Result<ScanResult> {
    let log = store.log(table);
    let st = log.read_state(version)?;
    let mut stats = ScanStats {
        files_total: st.files.len(),
        bytes_total: st.bytes(),
        ..Default::default()
    };
    // The time window is just a predicate on `ts`, so it gets the SAME
    // page-level pruning as any other column instead of stopping at
    // row-group granularity. `ts` is the sort key, so its pages are
    // perfectly ordered and this is where page pruning pays most.
    let mut all_preds: Vec<Predicate> = preds.to_vec();
    if let Some(t) = t0 {
        all_preds.push(Predicate {
            column: "ts".into(),
            op: crate::predicate::Op::Ge,
            value: crate::predicate::Value::Int(t),
        });
    }
    if let Some(t) = t1 {
        all_preds.push(Predicate {
            column: "ts".into(),
            op: crate::predicate::Op::Le,
            value: crate::predicate::Value::Int(t),
        });
    }
    let page_preds: &[Predicate] = &all_preds;

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
        // share this metadata instead of re-reading it. The page index is
        // loaded only when a predicate can actually use it — it is extra
        // footer bytes, and paying for it on an unfiltered scan would make
        // the elision number worse for no gain.
        let opts = ArrowReaderOptions::new().with_page_index_policy(
            if page_preds.is_empty() {
                PageIndexPolicy::Skip
            } else {
                PageIndexPolicy::Optional
            },
        );
        let meta = ArrowReaderMetadata::load(&cf, opts)?;
        let md = meta.metadata().clone();
        let descr = md.file_metadata().schema_descr();

        // ts leaf index (top-level i64 column named "ts")
        let ts_leaf = (0..descr.num_columns())
            .find(|&i| descr.column(i).path().parts()[0] == "ts");

        // layer 2: row-group pruning on footer statistics — the ts window
        // and every pushed predicate
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
            if overlap
                && predicate::row_group_may_match(md.row_group(rg), descr, preds)
            {
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

        // layer 3: page pruning inside each surviving row group
        let selections: Vec<Option<RowSelection>> = keep
            .iter()
            .map(|&rg| page_selection(&md, descr, rg, page_preds, &mut stats))
            .collect();

        // decode surviving row groups in parallel; each task gets its own
        // reader over the same counted file handle and shared footer
        let file_batches: Vec<Vec<RecordBatch>> = keep
            .par_iter()
            .zip(selections.into_par_iter())
            .map(|(&rg, sel)| -> Result<Vec<RecordBatch>> {
                let mut b = ParquetRecordBatchReaderBuilder::new_with_metadata(
                    cf.clone(),
                    meta.clone(),
                )
                .with_row_groups(vec![rg])
                .with_projection(mask.clone())
                .with_batch_size(65_536);
                if let Some(sel) = sel {
                    b = b.with_row_selection(sel);
                }
                // LATE MATERIALIZATION (C-Store's idea, and the reason
                // Lance-style random access matters here): evaluate the
                // predicates against ONLY their own columns first, then
                // read the wide columns for the surviving rows alone. On
                // a 1152-d vector table, `ts` is a rounding error and the
                // vectors are everything, so deciding which rows to want
                // before touching them is the whole game.
                if filter_pays(&md, rg, descr, page_preds, &mask) {
                    if let Some(filter) = row_filter(descr, page_preds) {
                        b = b.with_row_filter(filter);
                    }
                }
                let reader = b
                .build()?;
                let mut out = Vec::new();
                for batch in reader {
                    // the RowFilter already applied the predicates
                    // exactly; this stays as the guard for columns the
                    // filter could not be built for
                    let batch = filter_rows(filter_ts(batch?, t0, t1)?, preds)?;
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

/// Layer 3: turn the page index into a RowSelection so the reader skips
/// pages that cannot hold a match. Intersects the ranges of every
/// predicate (AND), and returns None when nothing can be pruned — a
/// full selection would only add work.
fn page_selection(
    md: &parquet::file::metadata::ParquetMetaData,
    descr: &parquet::schema::types::SchemaDescriptor,
    rg: usize,
    preds: &[Predicate],
    stats: &mut ScanStats,
) -> Option<RowSelection> {
    if preds.is_empty() {
        return None;
    }
    let col_idx = md.column_index()?;
    let off_idx = md.offset_index()?;
    let rg_rows = md.row_group(rg).num_rows() as usize;

    let mut keep: Option<Vec<(usize, usize)>> = None;
    let mut total_pages = 0usize;
    for p in preds {
        let leaf = predicate::leaf_of(descr, &p.column)?;
        let ci = col_idx.get(rg)?.get(leaf)?;
        let oi = off_idx.get(rg)?.get(leaf)?;
        total_pages = total_pages.max(oi.page_locations().len());
        let ranges = predicate::page_ranges(ci, oi, rg_rows, p)?;
        keep = Some(match keep {
            None => ranges,
            Some(prev) => intersect(&prev, &ranges),
        });
    }
    let keep = keep?;
    let kept_rows: usize = keep.iter().map(|(a, b)| b - a).sum();
    stats.pages_total += total_pages;
    // page count is per column; report the fraction of rows surviving in
    // page terms so the number stays meaningful across column counts
    stats.pages_scanned += if rg_rows == 0 {
        0
    } else {
        (total_pages * kept_rows).div_ceil(rg_rows.max(1))
    };
    if kept_rows == rg_rows {
        return None; // nothing pruned
    }

    let mut sel = Vec::new();
    let mut at = 0usize;
    for (a, b) in keep {
        if a > at {
            sel.push(RowSelector::skip(a - at));
        }
        sel.push(RowSelector::select(b - a));
        at = b;
    }
    if at < rg_rows {
        sel.push(RowSelector::skip(rg_rows - at));
    }
    Some(RowSelection::from(sel))
}

fn intersect(a: &[(usize, usize)], b: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        if lo < hi {
            out.push((lo, hi));
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// Is late materialization worth it here? The filter re-reads the
/// predicate columns (once to decide, once to output), so on a narrow
/// table it can read MORE than a plain scan — measured at -20% elision
/// on `frames` before this check existed. It only pays when the columns
/// being protected are much larger than the columns doing the deciding.
fn filter_pays(
    md: &parquet::file::metadata::ParquetMetaData,
    rgi: usize,
    descr: &parquet::schema::types::SchemaDescriptor,
    preds: &[Predicate],
    mask: &ProjectionMask,
) -> bool {
    if preds.is_empty() {
        return false;
    }
    let rg = md.row_group(rgi);
    let mut pred_bytes = 0i64;
    let mut proj_bytes = 0i64;
    let mut payload_pages = 0usize;
    for leaf in 0..descr.num_columns() {
        let sz = rg.column(leaf).compressed_size();
        let root = descr.column(leaf).path().parts()[0].clone();
        let is_pred = preds.iter().any(|p| p.column == root);
        if is_pred {
            pred_bytes += sz;
        }
        if mask.leaf_included(leaf) {
            proj_bytes += sz;
            if !is_pred {
                // how finely can rows be skipped in this column?
                payload_pages = payload_pages.max(
                    md.offset_index()
                        .and_then(|oi| oi.get(rgi))
                        .and_then(|cols| cols.get(leaf))
                        .map(|o| o.page_locations().len())
                        .unwrap_or(1),
                );
            }
        }
    }
    // A row selection can only skip whole PAGES. When the columns being
    // protected hold one page per row group there is nothing to skip,
    // and the filter just reads the predicate column twice — measured
    // at -20% elision on `frames` before this check.
    payload_pages > 1
        && proj_bytes > 0
        && (pred_bytes as f64) < 0.25 * (proj_bytes as f64)
}

/// Build a RowFilter that evaluates each predicate against its own
/// column only. Returns None when any predicate's column is missing —
/// the whole point is to avoid decoding wide columns, so a filter that
/// cannot be built cleanly is better skipped than half applied.
fn row_filter(
    descr: &parquet::schema::types::SchemaDescriptor,
    preds: &[Predicate],
) -> Option<RowFilter> {
    if preds.is_empty() {
        return None;
    }
    let mut fns: Vec<Box<dyn parquet::arrow::arrow_reader::ArrowPredicate>> =
        Vec::with_capacity(preds.len());
    for p in preds {
        let leaf = predicate::leaf_of(descr, &p.column)?;
        let mask = ProjectionMask::leaves(descr, [leaf]);
        let pred = p.clone();
        fns.push(Box::new(ArrowPredicateFn::new(mask, move |batch| {
            mask_of(&batch, &pred)
                .map_err(|e| arrow_schema::ArrowError::ComputeError(e.to_string()))
        })));
    }
    Some(RowFilter::new(fns))
}

/// The boolean mask a single predicate produces on a batch.
fn mask_of(
    batch: &RecordBatch,
    p: &Predicate,
) -> Result<arrow_array::BooleanArray> {
    use crate::predicate::Value;
    let i = batch
        .schema()
        .index_of(&p.column)
        .with_context(|| format!("no column {:?}", p.column))?;
    let col = batch.column(i);
    Ok(match &p.value {
        Value::Int(v) => {
            let c = arrow_cast::cast(col, &arrow_schema::DataType::Int64)?;
            cmp_op(p.op, &c, &Int64Array::new_scalar(*v))?
        }
        Value::Float(v) => {
            let c = arrow_cast::cast(col, &arrow_schema::DataType::Float64)?;
            cmp_op(p.op, &c, &arrow_array::Float64Array::new_scalar(*v))?
        }
        Value::Bool(v) => {
            cmp_op(p.op, col, &arrow_array::BooleanArray::new_scalar(*v))?
        }
        Value::Str(v) => {
            let c = arrow_cast::cast(col, &arrow_schema::DataType::Utf8)?;
            cmp_op(p.op, &c, &arrow_array::StringArray::new_scalar(v.clone()))?
        }
    })
}

/// Layer 4: exact evaluation of the predicates on decoded rows.
fn filter_rows(batch: RecordBatch, preds: &[Predicate]) -> Result<RecordBatch> {
    use crate::predicate::Value;
    if preds.is_empty() {
        return Ok(batch);
    }
    let mut mask: Option<arrow_array::BooleanArray> = None;
    for p in preds {
        let Ok(i) = batch.schema().index_of(&p.column) else {
            anyhow::bail!("no column {:?} in {:?}", p.column,
                          batch.schema().fields().iter()
                              .map(|f| f.name().clone()).collect::<Vec<_>>());
        };
        let col = batch.column(i);
        let m = match &p.value {
            Value::Int(v) => {
                let s = Int64Array::new_scalar(*v);
                let c = arrow_cast::cast(col, &arrow_schema::DataType::Int64)?;
                cmp_op(p.op, &c, &s)?
            }
            Value::Float(v) => {
                let s = arrow_array::Float64Array::new_scalar(*v);
                let c = arrow_cast::cast(col, &arrow_schema::DataType::Float64)?;
                cmp_op(p.op, &c, &s)?
            }
            Value::Bool(v) => {
                let s = arrow_array::BooleanArray::new_scalar(*v);
                cmp_op(p.op, col, &s)?
            }
            Value::Str(v) => {
                let s = arrow_array::StringArray::new_scalar(v.clone());
                let c = arrow_cast::cast(col, &arrow_schema::DataType::Utf8)?;
                cmp_op(p.op, &c, &s)?
            }
        };
        mask = Some(match mask {
            None => m,
            Some(prev) => arrow_arith::boolean::and(&prev, &m)?,
        });
    }
    Ok(arrow_select::filter::filter_record_batch(&batch, &mask.unwrap())?)
}

fn cmp_op(
    op: crate::predicate::Op,
    lhs: &dyn arrow_array::Datum,
    rhs: &dyn arrow_array::Datum,
) -> Result<arrow_array::BooleanArray> {
    use crate::predicate::Op;
    Ok(match op {
        Op::Eq => arrow_ord::cmp::eq(lhs, rhs)?,
        Op::Ne => arrow_ord::cmp::neq(lhs, rhs)?,
        Op::Lt => arrow_ord::cmp::lt(lhs, rhs)?,
        Op::Le => arrow_ord::cmp::lt_eq(lhs, rhs)?,
        Op::Gt => arrow_ord::cmp::gt(lhs, rhs)?,
        Op::Ge => arrow_ord::cmp::gt_eq(lhs, rhs)?,
    })
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
