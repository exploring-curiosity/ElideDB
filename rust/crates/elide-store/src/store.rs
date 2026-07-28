//! Store open + inventory: the fold of every table's log.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::log::{TableLog, TableState};

#[derive(Debug, Deserialize)]
pub struct StoreMeta {
    pub format: String,
    pub name: String,
    #[serde(default)]
    pub created_utc: String,
}

pub struct Store {
    pub root: PathBuf,
    pub meta: StoreMeta,
}

#[derive(Debug)]
pub struct TableSummary {
    pub name: String,
    pub state: TableState,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let meta_path = path.join("_store.json");
        let raw = fs::read_to_string(&meta_path)
            .with_context(|| format!("not an ElideDB store (no {meta_path:?})"))?;
        let meta: StoreMeta = serde_json::from_str(&raw)?;
        // "streetdex-lake/2" is the pre-rename spelling of the same format;
        // stores created before 2026-07-18 still carry it
        if !meta.format.starts_with("elidedb/")
            && !meta.format.starts_with("streetdex-lake/")
        {
            bail!("unknown store format {:?}", meta.format);
        }
        Ok(Self { root: path.to_path_buf(), meta })
    }

    pub fn table_dir(&self, name: &str) -> PathBuf {
        self.root.join("tables").join(name)
    }

    pub fn tables(&self) -> Result<Vec<String>> {
        let mut out = Vec::new();
        let dir = self.root.join("tables");
        let Ok(rd) = fs::read_dir(&dir) else { return Ok(out) };
        for e in rd {
            let e = e?;
            if e.path().join("_log").is_dir() {
                out.push(e.file_name().to_string_lossy().into_owned());
            }
        }
        out.sort();
        Ok(out)
    }

    pub fn log(&self, table: &str) -> TableLog {
        TableLog::new(&self.table_dir(table))
    }

    pub fn describe(&self, version: Option<u64>) -> Result<Vec<TableSummary>> {
        let mut out = Vec::new();
        for name in self.tables()? {
            let state = self.log(&name).read_state(version)?;
            out.push(TableSummary { name, state });
        }
        Ok(out)
    }
}
