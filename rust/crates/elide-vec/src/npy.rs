//! Minimal .npy reader for query vectors (little-endian f32, C order).
//! The ML sidecar writes query embeddings as .npy; this is the whole
//! interprocess contract for text queries — files, no RPC.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};

pub fn read_f32_1d(path: &Path) -> Result<Vec<f32>> {
    let raw = fs::read(path).with_context(|| format!("read {path:?}"))?;
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        bail!("{path:?} is not a .npy file");
    }
    let (header, data_off) = match raw[6] {
        1 => {
            let n = u16::from_le_bytes([raw[8], raw[9]]) as usize;
            (String::from_utf8_lossy(&raw[10..10 + n]).into_owned(), 10 + n)
        }
        2 | 3 => {
            let n = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
            (String::from_utf8_lossy(&raw[12..12 + n]).into_owned(), 12 + n)
        }
        v => bail!("unsupported npy version {v}"),
    };
    if !header.contains("'<f4'") {
        bail!("query vector must be little-endian float32, got header {header}");
    }
    if header.contains("'fortran_order': True") {
        bail!("fortran-order npy not supported");
    }
    let body = &raw[data_off..];
    if body.len() % 4 != 0 {
        bail!("truncated npy payload");
    }
    Ok(body
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}
