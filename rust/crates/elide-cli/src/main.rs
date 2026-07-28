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
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Stats { store, version, json } => stats(&store, version, json),
        Cmd::Scan { store, table, t0, t1, columns, version, json, repeat } => {
            let cols: Option<Vec<String>> = columns
                .map(|s| s.split(',').map(|c| c.trim().to_string()).collect());
            scan(&store, &table, t0, t1, cols.as_deref(), version, json, repeat)
        }
    }
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
) -> Result<()> {
    let store = Store::open(path)?;
    let mut lat_ms = Vec::with_capacity(repeat as usize);
    let mut r = elide_store::scan(&store, table, t0, t1, columns, version)?;
    for _ in 1..repeat {
        let t = std::time::Instant::now();
        r = elide_store::scan(&store, table, t0, t1, columns, version)?;
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
        "rows {}   bytes read {} / {}   elided {:.2}%   files {}/{}   row groups {}/{}",
        r.stats.rows,
        r.stats.bytes_read,
        r.stats.bytes_total,
        r.stats.elided_pct(),
        r.stats.files_scanned,
        r.stats.files_total,
        r.stats.row_groups_scanned,
        r.stats.row_groups_total,
    );
    Ok(())
}
