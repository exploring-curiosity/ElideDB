//! Exact and tiered vector search over a store table.

use anyhow::{bail, Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use elide_store::Store;
use rayon::prelude::*;

use crate::codes::CodesFile;

/// A vector table materialized in scan order: ids stay implicit (row i of
/// the version-N scan), keys carry (ts, stream) for reporting results.
pub struct VecTable {
    pub version: u64,
    pub n: usize,
    pub dim: usize,
    /// row-major, L2-normalized
    pub data: Vec<f32>,
    pub ts: Vec<i64>,
    pub stream: Vec<String>,
    pub bytes_read: u64,
}

impl VecTable {
    pub fn load(store: &Store, table: &str, version: Option<u64>) -> Result<Self> {
        let cols = ["stream".to_string(), "vector".to_string()];
        let r = elide_store::scan(store, table, None, None, Some(&cols), version)?;
        let ver = store.log(table).read_state(version)?.version;
        let mut data = Vec::new();
        let mut ts = Vec::new();
        let mut stream = Vec::new();
        let mut dim = 0usize;
        for batch in &r.batches {
            let ts_col = batch
                .column(batch.schema().index_of("ts")?)
                .as_primitive::<arrow_array::types::Int64Type>();
            ts.extend(ts_col.values().iter().copied());
            let st_col = batch
                .column(batch.schema().index_of("stream")?)
                .as_string::<i32>();
            stream.extend(st_col.iter().map(|s| s.unwrap_or("").to_string()));
            let vec_col = batch
                .column(batch.schema().index_of("vector")?)
                .as_fixed_size_list();
            dim = vec_col.value_length() as usize;
            // fp32 (legacy) or fp16 (post-compress) storage; search math is
            // fp32 either way
            match vec_col.values().data_type() {
                arrow_schema::DataType::Float32 => {
                    let vals = vec_col.values().as_primitive::<Float32Type>();
                    data.extend(vals.values().iter().copied());
                }
                arrow_schema::DataType::Float16 => {
                    let vals = vec_col
                        .values()
                        .as_primitive::<arrow_array::types::Float16Type>();
                    data.extend(vals.values().iter().map(|v| v.to_f32()));
                }
                other => bail!("unsupported vector element type {other:?}"),
            }
        }
        let n = ts.len();
        if n == 0 {
            bail!("{table} has no vectors");
        }
        // L2-normalize once so every later score is a plain dot product
        data.par_chunks_mut(dim).for_each(|row| {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                row.iter_mut().for_each(|v| *v /= norm);
            }
        });
        Ok(Self { version: ver, n, dim, data, ts, stream, bytes_read: r.stats.bytes_read })
    }
}

pub fn normalize(q: &mut [f32]) {
    let n = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n > 0.0 {
        q.iter_mut().for_each(|v| *v /= n);
    }
}

fn top_k_of(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
        scores[b].total_cmp(&scores[a])
    });
    let mut top: Vec<(usize, f32)> = idx[..k].iter().map(|&i| (i, scores[i])).collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    top
}

/// Exact cosine top-k: parallel dot products over the full matrix.
pub fn exact_top_k(t: &VecTable, q: &[f32], k: usize) -> Vec<(usize, f32)> {
    let scores: Vec<f32> = t
        .data
        .par_chunks(t.dim)
        .map(|row| row.iter().zip(q).map(|(a, b)| a * b).sum())
        .collect();
    top_k_of(&scores, k)
}

/// Tiered top-k: Hamming scan over mmap'd sign codes -> shortlist ->
/// exact rerank of the shortlist only. Returns hits plus the bytes the
/// tier actually touched (codes + reranked rows), the honest cost.
pub fn tiered_top_k(
    t: &VecTable,
    codes: &CodesFile,
    q: &[f32],
    k: usize,
    shortlist: usize,
) -> Result<(Vec<(usize, f32)>, u64)> {
    if codes.n != t.n || codes.dim != t.dim {
        bail!(
            "codes file is stale: codes n={} dim={} vs table n={} dim={} — \
             rebuild with `elide vindex`",
            codes.n, codes.dim, t.n, t.dim
        );
    }
    let qc = codes.encode_query(q);
    let dists: Vec<u32> = (0..codes.n)
        .into_par_iter()
        .map(|i| codes.hamming(i, &qc))
        .collect();
    let m = shortlist.min(codes.n);
    let mut idx: Vec<usize> = (0..codes.n).collect();
    idx.select_nth_unstable_by_key(m.saturating_sub(1), |&i| dists[i]);
    let short = &idx[..m];
    let scores: Vec<(usize, f32)> = short
        .iter()
        .map(|&i| {
            let row = &t.data[i * t.dim..(i + 1) * t.dim];
            (i, row.iter().zip(q).map(|(a, b)| a * b).sum())
        })
        .collect();
    let mut top = scores;
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    top.truncate(k);
    let bytes = (codes.n * codes.lanes * 8 + m * t.dim * 4) as u64;
    Ok((top, bytes))
}

/// Leave-one-out recall of the tiered path vs exact, on `queries` sampled
/// table rows — the acceptance gate a codes artifact must pass before the
/// tier is trusted (>= 0.98 @ 10 per docs/ENGINE.md).
pub fn self_test(
    t: &VecTable,
    codes: &CodesFile,
    queries: usize,
    k: usize,
    shortlist: usize,
) -> Result<f64> {
    let step = (t.n / queries.min(t.n)).max(1);
    let mut recalls = Vec::new();
    for qi in (0..t.n).step_by(step).take(queries) {
        let q: Vec<f32> = t.data[qi * t.dim..(qi + 1) * t.dim].to_vec();
        let exact: Vec<usize> = exact_top_k(t, &q, k + 1)
            .into_iter()
            .map(|(i, _)| i)
            .filter(|&i| i != qi)
            .take(k)
            .collect();
        let (tiered, _) = tiered_top_k(t, codes, &q, k + 1, shortlist)?;
        let tiered: std::collections::HashSet<usize> = tiered
            .into_iter()
            .map(|(i, _)| i)
            .filter(|&i| i != qi)
            .take(k)
            .collect();
        let hit = exact.iter().filter(|i| tiered.contains(i)).count();
        recalls.push(hit as f64 / exact.len().max(1) as f64);
    }
    Ok(recalls.iter().sum::<f64>() / recalls.len().max(1) as f64)
}

pub fn build_for(store: &Store, table: &str) -> Result<(std::path::PathBuf, u64, u64)> {
    let t = VecTable::load(store, table, None).context("load vectors")?;
    let path = crate::codes::artifact_path(&store.table_dir(table), t.version);
    let bytes = crate::codes::build_codes(&t.data, t.n, t.dim, &path)?;
    Ok((path, bytes, t.version))
}
