//! The scan-tier sidecar: 1-bit sign codes, u64-lane padded, mmap-able.
//!
//! `tables/<table>/_codes.v<N>.bin` where N is the table's DATA version —
//! derived artifacts bind to the data they were built from and never touch
//! the log (the v1 lesson: an index build that bumps the version
//! self-invalidates). Row i of the codes is row i of the version-N scan in
//! its deterministic ts-sorted order.
//!
//! Layout (little-endian):
//!   0   4  magic "EVC1"
//!   4   4  u32 layout version = 1
//!   8   4  u32 codec: 1 = sign-v1 (bit = value >= 0, unrotated)
//!   12  4  u32 dim
//!   16  8  u64 n rows
//!   24  4  u32 code_bytes per row (ceil(dim/64)*8, u64 aligned)
//!   28  4  reserved
//!   32  ..  codes, row-major

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use memmap2::Mmap;

pub const CODEC_SIGN_V1: u32 = 1;
const HEADER: usize = 32;

pub fn artifact_path(table_dir: &Path, version: u64) -> PathBuf {
    table_dir.join(format!("_codes.v{version}.bin"))
}

pub fn build_codes(vectors: &[f32], n: usize, dim: usize, out: &Path) -> Result<u64> {
    let lanes = dim.div_ceil(64);
    let code_bytes = lanes * 8;
    let mut buf = Vec::with_capacity(HEADER + n * code_bytes);
    buf.extend_from_slice(b"EVC1");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&CODEC_SIGN_V1.to_le_bytes());
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    buf.extend_from_slice(&(n as u64).to_le_bytes());
    buf.extend_from_slice(&(code_bytes as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    for row in 0..n {
        let v = &vectors[row * dim..(row + 1) * dim];
        for lane in 0..lanes {
            let mut w = 0u64;
            for bit in 0..64 {
                let i = lane * 64 + bit;
                if i < dim && v[i] >= 0.0 {
                    w |= 1 << bit;
                }
            }
            buf.extend_from_slice(&w.to_le_bytes());
        }
    }
    let tmp = out.with_extension("tmp");
    let mut f = fs::File::create(&tmp)?;
    f.write_all(&buf)?;
    f.sync_all()?;
    fs::rename(&tmp, out)?;
    Ok(buf.len() as u64)
}

pub struct CodesFile {
    mmap: Mmap,
    pub n: usize,
    pub dim: usize,
    pub lanes: usize,
}

impl CodesFile {
    pub fn open(path: &Path) -> Result<Self> {
        let f = fs::File::open(path).with_context(|| format!("open {path:?}"))?;
        let mmap = unsafe { Mmap::map(&f)? };
        if mmap.len() < HEADER || &mmap[..4] != b"EVC1" {
            bail!("{path:?} is not an EVC1 codes file");
        }
        let codec = u32::from_le_bytes(mmap[8..12].try_into()?);
        if codec != CODEC_SIGN_V1 {
            bail!("unknown codes codec {codec}");
        }
        let dim = u32::from_le_bytes(mmap[12..16].try_into()?) as usize;
        let n = u64::from_le_bytes(mmap[16..24].try_into()?) as usize;
        let code_bytes = u32::from_le_bytes(mmap[24..28].try_into()?) as usize;
        let lanes = code_bytes / 8;
        if mmap.len() < HEADER + n * code_bytes {
            bail!("codes file truncated");
        }
        Ok(Self { mmap, n, dim, lanes })
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[u8] {
        let cb = self.lanes * 8;
        &self.mmap[HEADER + i * cb..HEADER + (i + 1) * cb]
    }

    pub fn encode_query(&self, q: &[f32]) -> Vec<u64> {
        let mut out = vec![0u64; self.lanes];
        for (i, &v) in q.iter().enumerate() {
            if v >= 0.0 {
                out[i / 64] |= 1 << (i % 64);
            }
        }
        out
    }

    #[inline]
    pub fn hamming(&self, i: usize, q: &[u64]) -> u32 {
        let row = self.row(i);
        let mut d = 0u32;
        for (lane, &qw) in q.iter().enumerate() {
            let w = u64::from_le_bytes(row[lane * 8..lane * 8 + 8].try_into().unwrap());
            d += (w ^ qw).count_ones();
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_hamming() {
        let dim = 130; // exercises the partial last lane
        let vecs: Vec<f32> = (0..3 * dim)
            .map(|i| if (i * 7) % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        let tmp = std::env::temp_dir().join("evc1_test.bin");
        build_codes(&vecs, 3, dim, &tmp).unwrap();
        let c = CodesFile::open(&tmp).unwrap();
        assert_eq!((c.n, c.dim, c.lanes), (3, dim, 3));
        // a row against its own code has hamming distance 0
        for row in 0..3 {
            let q = c.encode_query(&vecs[row * dim..(row + 1) * dim]);
            assert_eq!(c.hamming(row, &q), 0);
        }
        // flipping one dimension moves the distance by exactly 1
        let mut v: Vec<f32> = vecs[..dim].to_vec();
        v[129] = -v[129];
        let q = c.encode_query(&v);
        assert_eq!(c.hamming(0, &q), 1);
        std::fs::remove_file(tmp).ok();
    }
}
