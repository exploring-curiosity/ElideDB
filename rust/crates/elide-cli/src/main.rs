use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use elide_store::Store;

#[derive(Parser)]
#[command(name = "elide", about = "ElideDB engine (Rust core)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Table inventory from the transaction log (rows, bytes, ts range)
    Stats {
        store: PathBuf,
        #[arg(long)]
        version: Option<u64>,
        #[arg(long)]
        json: bool,
    },
    /// Counted, pruned time-range scan of one table
    Scan {
        store: PathBuf,
        table: String,
        #[arg(long)]
        t0: Option<i64>,
        #[arg(long)]
        t1: Option<i64>,
        /// comma-separated column names (ts always included)
        #[arg(long)]
        columns: Option<String>,
        #[arg(long)]
        version: Option<u64>,
        #[arg(long)]
        json: bool,
        /// run the scan N times in-process and report per-iteration latency
        /// (isolates engine time from process spawn)
        #[arg(long, default_value_t = 1)]
        repeat: u32,
        /// predicate pushed into the scan, e.g. --where "keyframe = true"
        /// (repeatable; all must hold)
        #[arg(long = "where", value_name = "PREDICATE")]
        wheres: Vec<String>,
    },
    /// Build the scan-tier codes sidecar for a vector table
    Vindex { store: PathBuf, table: String },
    /// Vector search: tiered (codes -> rerank) by default, --exact for flat
    Vsearch {
        store: PathBuf,
        table: String,
        /// query vector as .npy (little-endian float32)
        #[arg(long)]
        query_npy: PathBuf,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 100)]
        shortlist: usize,
        #[arg(long)]
        exact: bool,
        #[arg(long, default_value_t = 1)]
        repeat: u32,
    },
    /// Recall gate: tiered vs exact on sampled table rows (leave-one-out)
    Vselftest {
        store: PathBuf,
        table: String,
        #[arg(long, default_value_t = 200)]
        queries: usize,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 100)]
        shortlist: usize,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Stats { store, version, json } => stats(&store, version, json),
        Cmd::Scan { store, table, t0, t1, columns, version, json, repeat, wheres } => {
            let cols: Option<Vec<String>> = columns
                .map(|s| s.split(',').map(|c| c.trim().to_string()).collect());
            let preds = wheres
                .iter()
                .map(|w| elide_store::Predicate::parse(w))
                .collect::<Result<Vec<_>>>()?;
            scan(&store, &table, t0, t1, cols.as_deref(), version, json, repeat,
                 &preds)
        }
        Cmd::Vindex { store, table } => {
            let s = Store::open(&store)?;
            let (path, bytes, version) = elide_vec::build_for(&s, &table)?;
            println!("built {path:?}  ({bytes} bytes, data version {version})");
            Ok(())
        }
        Cmd::Vsearch { store, table, query_npy, k, shortlist, exact, repeat } => {
            vsearch(&store, &table, &query_npy, k, shortlist, exact, repeat)
        }
        Cmd::Vselftest { store, table, queries, k, shortlist } => {
            let s = Store::open(&store)?;
            let t = elide_vec::VecTable::load(&s, &table, None)?;
            let codes = elide_vec::CodesFile::open(&elide_vec::artifact_path(
                &s.table_dir(&table),
                t.version,
            ))?;
            let recall = elide_vec::self_test(&t, &codes, queries, k, shortlist)?;
            println!(
                "recall@{k} = {recall:.4}  ({queries} queries, shortlist {shortlist}, \
                 n={}, dim={})",
                t.n, t.dim
            );
            Ok(())
        }
    }
}

fn vsearch(
    store: &PathBuf,
    table: &str,
    query_npy: &PathBuf,
    k: usize,
    shortlist: usize,
    exact: bool,
    repeat: u32,
) -> Result<()> {
    let s = Store::open(store)?;
    let t = elide_vec::VecTable::load(&s, table, None)?;
    let mut q = elide_vec::npy::read_f32_1d(query_npy)?;
    anyhow::ensure!(
        q.len() == t.dim,
        "query dim {} != table dim {}",
        q.len(),
        t.dim
    );
    elide_vec::normalize(&mut q);
    let mut lat_ms = Vec::new();
    let (hits, tier_bytes) = if exact {
        let mut hits = elide_vec::exact_top_k(&t, &q, k);
        for _ in 1..repeat {
            let s0 = std::time::Instant::now();
            hits = elide_vec::exact_top_k(&t, &q, k);
            lat_ms.push(s0.elapsed().as_secs_f64() * 1e3);
        }
        (hits, (t.n * t.dim * 4) as u64)
    } else {
        let codes = elide_vec::CodesFile::open(&elide_vec::artifact_path(
            &s.table_dir(table),
            t.version,
        ))?;
        let (mut hits, mut bytes) = elide_vec::tiered_top_k(&t, &codes, &q, k, shortlist)?;
        for _ in 1..repeat {
            let s0 = std::time::Instant::now();
            (hits, bytes) = elide_vec::tiered_top_k(&t, &codes, &q, k, shortlist)?;
            lat_ms.push(s0.elapsed().as_secs_f64() * 1e3);
        }
        (hits, bytes)
    };
    for (i, score) in &hits {
        println!("{:.4}  ts {}  stream {}", score, t.ts[*i], t.stream[*i]);
    }
    println!(
        "tier bytes touched {} of {} fp32 bytes ({:.1}x less)",
        tier_bytes,
        t.n * t.dim * 4,
        (t.n * t.dim * 4) as f64 / tier_bytes.max(1) as f64
    );
    if !lat_ms.is_empty() {
        lat_ms.sort_by(|a, b| a.total_cmp(b));
        println!(
            "in-process lat p50 {:.3} ms  max {:.3} ms",
            lat_ms[lat_ms.len() / 2],
            lat_ms[lat_ms.len() - 1]
        );
    }
    Ok(())
}

fn stats(path: &PathBuf, version: Option<u64>, json: bool) -> Result<()> {
    let store = Store::open(path)?;
    let tables = store.describe(version)?;
    if json {
        let out: Vec<_> = tables
            .iter()
            .map(|t| {
                serde_json::json!({
                    "table": t.name,
                    "version": t.state.version,
                    "kind": t.state.kind,
                    "files": t.state.files.len(),
                    "rows": t.state.rows(),
                    "bytes": t.state.bytes(),
                    "min_ts": t.state.min_ts(),
                    "max_ts": t.state.max_ts(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }
    println!("store: {}  ({})", store.meta.name, store.meta.format);
    println!(
        "{:<22} {:>4} {:>6} {:>12} {:>14}  ts range",
        "table", "ver", "files", "rows", "bytes"
    );
    for t in &tables {
        println!(
            "{:<22} {:>4} {:>6} {:>12} {:>14}  [{} .. {}]",
            t.name,
            t.state.version,
            t.state.files.len(),
            t.state.rows(),
            t.state.bytes(),
            t.state.min_ts(),
            t.state.max_ts()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scan(
    path: &PathBuf,
    table: &str,
    t0: Option<i64>,
    t1: Option<i64>,
    columns: Option<&[String]>,
    version: Option<u64>,
    json: bool,
    repeat: u32,
    preds: &[elide_store::Predicate],
) -> Result<()> {
    let store = Store::open(path)?;
    let mut lat_ms = Vec::with_capacity(repeat as usize);
    let mut r =
        elide_store::scan_where(&store, table, t0, t1, columns, version, preds)?;
    for _ in 1..repeat {
        let t = std::time::Instant::now();
        r = elide_store::scan_where(&store, table, t0, t1, columns, version,
                                    preds)?;
        lat_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    if json {
        let mut out = serde_json::to_value(&r.stats)?;
        if !lat_ms.is_empty() {
            lat_ms.sort_by(|a, b| a.total_cmp(b));
            out["lat_ms_p50"] = serde_json::json!(lat_ms[lat_ms.len() / 2]);
            out["lat_ms_max"] = serde_json::json!(lat_ms[lat_ms.len() - 1]);
        }
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }
    println!(
        "rows {}   bytes read {} / {}   elided {:.2}%   files {}/{}   \
         row groups {}/{}   pages ~{}/{}",
        r.stats.rows,
        r.stats.bytes_read,
        r.stats.bytes_total,
        r.stats.elided_pct(),
        r.stats.files_scanned,
        r.stats.files_total,
        r.stats.row_groups_scanned,
        r.stats.row_groups_total,
        r.stats.pages_scanned,
        r.stats.pages_total,
    );
    Ok(())
}
