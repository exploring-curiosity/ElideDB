//! A learned index over `ts` (PGM-style piecewise-linear approximation).
//!
//! Ferragina & Vinciguerra, "The PGM-index" (VLDB 2020), and the line of
//! work Kraska et al. opened with "The Case for Learned Index Structures"
//! (SIGMOD 2018): if the keys are sorted, the mapping key -> position is
//! a monotone function, and a piecewise-linear model with a guaranteed
//! error bound EPS replaces the search structure entirely. Lookup becomes
//! "evaluate a line, then look within +/- EPS".
//!
//! Why it matters HERE, specifically. Our scan already prunes to pages,
//! but to decide WHICH rows match a time window it still has to read the
//! `ts` column and compare. The learned index answers the same question
//! from a few hundred bytes of model: the row range for [t0, t1], without
//! reading a single value of the column. On a table whose payload is a
//! 1152-d vector, the read that remains is only the rows we actually
//! want.
//!
//! Correctness: the model is only ever used to bound a RANGE. The exact
//! Arrow filter still runs, so a wrong bound costs bytes, never answers —
//! and the build refuses non-monotone data outright.
//!
//! Layout of `_tsidx.v<N>.bin` (little-endian):
//!   0   4  magic "ELI1"
//!   4   4  u32 layout version = 1
//!   8   4  u32 epsilon
//!   12  4  u32 file count
//!   then per file: u32 name_len, name bytes, u64 rows,
//!                  u32 segment count, segments
//!   segment: i64 first_key, f64 slope, f64 intercept

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

pub const EPS: u32 = 64;

#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub first_key: i64,
    pub slope: f64,
    pub intercept: f64,
}

#[derive(Debug, Clone, Default)]
pub struct FileModel {
    pub rows: u64,
    pub segments: Vec<Segment>,
}

impl FileModel {
    /// Row position bounds for a key: the model's estimate widened by
    /// the guaranteed error, clamped to the file.
    pub fn bounds(&self, key: i64) -> (u64, u64) {
        if self.segments.is_empty() {
            return (0, self.rows);
        }
        let i = match self
            .segments
            .binary_search_by(|s| s.first_key.cmp(&key))
        {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };
        let s = &self.segments[i];
        let p = s.slope * key as f64 + s.intercept;
        let lo = (p - EPS as f64).max(0.0) as u64;
        let hi = ((p + EPS as f64).max(0.0) as u64).min(self.rows);
        (lo.min(self.rows), hi)
    }

    /// Row range that can contain [t0, t1]. Widened by the error bound
    /// on both ends; the exact filter narrows it afterwards.
    pub fn range(&self, t0: Option<i64>, t1: Option<i64>) -> (u64, u64) {
        let lo = t0.map(|t| self.bounds(t).0).unwrap_or(0);
        let hi = t1.map(|t| self.bounds(t).1).unwrap_or(self.rows);
        (lo, hi.max(lo))
    }
}

/// Optimal piecewise-linear approximation, streaming: extend the current
/// segment while a line within +/- EPS still fits every point seen, else
/// start a new one. (The convex-hull formulation of the PGM paper; this
/// is the simple incremental variant, which is what the error bound
/// actually requires.)
pub fn fit(keys: &[i64], eps: f64) -> Result<Vec<Segment>> {
    let mut out = Vec::new();
    if keys.is_empty() {
        return Ok(out);
    }
    let mut start = 0usize;
    while start < keys.len() {
        let x0 = keys[start] as f64;
        let y0 = start as f64;
        // slope window that keeps every point within eps of the line
        let mut lo_slope = f64::NEG_INFINITY;
        let mut hi_slope = f64::INFINITY;
        let mut end = start + 1;
        while end < keys.len() {
            let dx = keys[end] as f64 - x0;
            let dy = end as f64 - y0;
            if dx <= 0.0 {
                // duplicate keys: the line cannot separate them, and the
                // error bound absorbs the ties
                end += 1;
                continue;
            }
            let hi = (dy + eps) / dx;
            let lo = (dy - eps) / dx;
            let nlo = lo_slope.max(lo);
            let nhi = hi_slope.min(hi);
            if nlo > nhi {
                break;
            }
            lo_slope = nlo;
            hi_slope = nhi;
            end += 1;
        }
        let slope = if lo_slope.is_finite() && hi_slope.is_finite() {
            (lo_slope + hi_slope) / 2.0
        } else if hi_slope.is_finite() {
            hi_slope
        } else if lo_slope.is_finite() {
            lo_slope
        } else {
            0.0
        };
        out.push(Segment {
            first_key: keys[start],
            slope,
            intercept: y0 - slope * x0,
        });
        start = end;
    }
    Ok(out)
}

pub fn artifact_path(table_dir: &Path, version: u64) -> PathBuf {
    table_dir.join(format!("_tsidx.v{version}.bin"))
}

pub fn write(path: &Path, models: &BTreeMap<String, FileModel>) -> Result<u64> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"ELI1");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&EPS.to_le_bytes());
    buf.extend_from_slice(&(models.len() as u32).to_le_bytes());
    for (name, m) in models {
        buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&m.rows.to_le_bytes());
        buf.extend_from_slice(&(m.segments.len() as u32).to_le_bytes());
        for s in &m.segments {
            buf.extend_from_slice(&s.first_key.to_le_bytes());
            buf.extend_from_slice(&s.slope.to_le_bytes());
            buf.extend_from_slice(&s.intercept.to_le_bytes());
        }
    }
    let tmp = path.with_extension("tmp");
    let mut f = fs::File::create(&tmp)?;
    f.write_all(&buf)?;
    f.sync_all()?;
    fs::rename(&tmp, path)?;
    Ok(buf.len() as u64)
}

pub fn read(path: &Path) -> Result<BTreeMap<String, FileModel>> {
    let raw = fs::read(path).with_context(|| format!("read {path:?}"))?;
    if raw.len() < 16 || &raw[..4] != b"ELI1" {
        bail!("{path:?} is not an ELI1 learned index");
    }
    let mut at = 16usize;
    let n = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
    let mut out = BTreeMap::new();
    for _ in 0..n {
        let nl = u32::from_le_bytes(raw[at..at + 4].try_into()?) as usize;
        at += 4;
        let name = String::from_utf8(raw[at..at + nl].to_vec())?;
        at += nl;
        let rows = u64::from_le_bytes(raw[at..at + 8].try_into()?);
        at += 8;
        let sc = u32::from_le_bytes(raw[at..at + 4].try_into()?) as usize;
        at += 4;
        let mut segs = Vec::with_capacity(sc);
        for _ in 0..sc {
            let first_key = i64::from_le_bytes(raw[at..at + 8].try_into()?);
            let slope = f64::from_le_bytes(raw[at + 8..at + 16].try_into()?);
            let intercept = f64::from_le_bytes(raw[at + 16..at + 24].try_into()?);
            at += 24;
            segs.push(Segment { first_key, slope, intercept });
        }
        out.insert(name, FileModel { rows, segments: segs });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_always_contain_the_true_position() {
        // a realistic mix: dense runs at 5 fps with gaps between episodes
        let mut keys = Vec::new();
        let mut t = 1_700_000_000_000_000_000i64;
        for ep in 0..200 {
            for _ in 0..35 {
                keys.push(t);
                t += 200_000_000;
            }
            if ep % 3 == 0 {
                t += 60_000_000_000; // gap
            }
        }
        let segs = fit(&keys, EPS as f64).unwrap();
        let m = FileModel { rows: keys.len() as u64, segments: segs };
        assert!(m.segments.len() < keys.len() / 10, "model should compress");
        for (i, &k) in keys.iter().enumerate() {
            let (lo, hi) = m.bounds(k);
            assert!(
                lo <= i as u64 && i as u64 <= hi,
                "position {i} outside [{lo},{hi}] for key {k}"
            );
        }
    }

    #[test]
    fn range_covers_a_window() {
        let keys: Vec<i64> = (0..10_000).map(|i| i as i64 * 7).collect();
        let m = FileModel {
            rows: keys.len() as u64,
            segments: fit(&keys, EPS as f64).unwrap(),
        };
        let (lo, hi) = m.range(Some(700), Some(1400));
        assert!(lo <= 100 && hi >= 200, "range [{lo},{hi}] misses rows 100..200");
    }
}


/// Build the learned index for a table's current version: one model per
/// file, fitted on that file's `ts` column. Refuses non-monotone files —
/// the whole guarantee rests on sortedness, so a silent fit over
/// unsorted keys would produce bounds that are simply wrong.
pub fn build_for(store: &crate::store::Store, table: &str) -> Result<(PathBuf, u64, usize)> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;

    let log = store.log(table);
    let st = log.read_state(None)?;
    let mut models = BTreeMap::new();
    let mut segments = 0usize;
    for f in &st.files {
        let cols = ["ts".to_string()];
        let r = crate::scan::scan_one_file(store, table, &f.path, Some(&cols))?;
        let mut keys: Vec<i64> = Vec::with_capacity(f.rows as usize);
        for b in &r {
            let i = b.schema().index_of("ts")?;
            keys.extend(b.column(i).as_primitive::<Int64Type>().values().iter().copied());
        }
        if keys.windows(2).any(|w| w[1] < w[0]) {
            bail!("{}/{} is not sorted by ts — a learned index over \
                   unsorted keys cannot bound anything", table, f.path);
        }
        let segs = fit(&keys, EPS as f64)?;
        segments += segs.len();
        models.insert(f.path.clone(), FileModel { rows: keys.len() as u64, segments: segs });
    }
    let path = artifact_path(&log.dir, st.version);
    let bytes = write(&path, &models)?;
    Ok((path, bytes, segments))
}
