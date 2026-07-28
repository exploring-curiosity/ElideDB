//! Transaction log fold — the Rust twin of python/elidedb/log.py.
//!
//! The store format is language-neutral by contract: this module reads the
//! SAME `_log/*.json` commits and `*.checkpoint.json` files the Python
//! engine writes, with identical fold semantics (checkpoint seed, then the
//! tail of commits; a present `schema`/`table_kind` key replaces state even
//! when empty, mirroring Python's `entry.get(k, current)`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub rows: u64,
    pub bytes: u64,
    pub min_ts: i64,
    pub max_ts: i64,
}

#[derive(Debug, Clone, Default)]
pub struct TableState {
    pub version: u64,
    pub kind: String,
    pub schema: String,
    pub files: Vec<FileEntry>,
    pub meta: BTreeMap<String, Value>,
}

impl TableState {
    pub fn rows(&self) -> u64 {
        self.files.iter().map(|f| f.rows).sum()
    }
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|f| f.bytes).sum()
    }
    pub fn min_ts(&self) -> i64 {
        self.files.iter().map(|f| f.min_ts).min().unwrap_or(0)
    }
    pub fn max_ts(&self) -> i64 {
        self.files.iter().map(|f| f.max_ts).max().unwrap_or(0)
    }
}

#[derive(Deserialize)]
struct Commit {
    version: u64,
    table_kind: Option<String>,
    schema: Option<String>,
    #[serde(default)]
    add: Vec<FileEntry>,
    #[serde(default)]
    remove: Vec<String>,
    #[serde(default)]
    meta: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct Checkpoint {
    version: u64,
    kind: String,
    schema: String,
    files: Vec<FileEntry>,
    #[serde(default)]
    meta: BTreeMap<String, Value>,
}

pub struct TableLog {
    pub dir: PathBuf,
    log_dir: PathBuf,
}

impl TableLog {
    pub fn new(table_dir: &Path) -> Self {
        Self { dir: table_dir.to_path_buf(), log_dir: table_dir.join("_log") }
    }

    /// Commit versions present on disk, ascending. Commit files are the
    /// all-digit stems; checkpoints ("N.checkpoint.json") and crash-leftover
    /// ".tmp" files fall out of the digit test naturally.
    pub fn versions(&self) -> Result<Vec<u64>> {
        let mut out = Vec::new();
        let Ok(rd) = fs::read_dir(&self.log_dir) else { return Ok(out) };
        for e in rd {
            let name = e?.file_name();
            let name = name.to_string_lossy();
            if let Some(stem) = name.strip_suffix(".json") {
                if !stem.is_empty() && stem.bytes().all(|b| b.is_ascii_digit()) {
                    out.push(stem.parse()?);
                }
            }
        }
        out.sort_unstable();
        Ok(out)
    }

    fn checkpoints(&self) -> Result<Vec<u64>> {
        let mut out = Vec::new();
        let Ok(rd) = fs::read_dir(&self.log_dir) else { return Ok(out) };
        for e in rd {
            let name = e?.file_name();
            let name = name.to_string_lossy();
            if let Some(stem) = name.strip_suffix(".checkpoint.json") {
                if !stem.is_empty() && stem.bytes().all(|b| b.is_ascii_digit()) {
                    out.push(stem.parse()?);
                }
            }
        }
        out.sort_unstable();
        Ok(out)
    }

    pub fn read_state(&self, version: Option<u64>) -> Result<TableState> {
        let mut st = TableState { kind: "timeseries".into(), ..Default::default() };
        let mut start = 0u64;
        for &v in self.checkpoints()?.iter().rev() {
            if version.is_none() || v <= version.unwrap() {
                let p = self.log_dir.join(format!("{v:020}.checkpoint.json"));
                let c: Checkpoint = serde_json::from_str(&fs::read_to_string(&p)?)
                    .with_context(|| format!("bad checkpoint {p:?}"))?;
                st.version = c.version;
                st.kind = c.kind;
                st.schema = c.schema;
                st.meta = c.meta;
                st.files = c.files;
                start = v;
                break;
            }
        }
        for v in self.versions()? {
            if v <= start {
                continue;
            }
            if let Some(cap) = version {
                if v > cap {
                    break;
                }
            }
            let p = self.log_dir.join(format!("{v:020}.json"));
            let e: Commit = serde_json::from_str(&fs::read_to_string(&p)?)
                .with_context(|| format!("bad commit {p:?}"))?;
            st.version = e.version;
            if let Some(k) = e.table_kind {
                st.kind = k;
            }
            if let Some(s) = e.schema {
                st.schema = s;
            }
            st.meta.extend(e.meta);
            if !e.remove.is_empty() {
                let removed: std::collections::HashSet<_> = e.remove.iter().collect();
                st.files.retain(|f| !removed.contains(&f.path));
            }
            st.files.extend(e.add);
        }
        Ok(st)
    }
}
