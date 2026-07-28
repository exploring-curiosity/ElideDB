//! Counted I/O — the founding metric as a type-level guarantee.
//!
//! Every byte the engine reads flows through a `CountingFile`; the scan API
//! accepts nothing else. Unlike the Python engine (which *estimates* touched
//! bytes from Parquet metadata), the counter here increments inside the read
//! calls themselves, so the elision number is what actually hit the file.

use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use parquet::errors::ParquetError;
use parquet::file::reader::{ChunkReader, Length};

#[derive(Clone, Default)]
pub struct ByteCounter(Arc<AtomicU64>);

impl ByteCounter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Clone)]
pub struct CountingFile {
    file: Arc<File>,
    len: u64,
    counter: ByteCounter,
}

impl CountingFile {
    pub fn open(path: &Path, counter: ByteCounter) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self { file: Arc::new(file), len, counter })
    }
}

impl Length for CountingFile {
    fn len(&self) -> u64 {
        self.len
    }
}

/// Positional reader: every read is a `pread` at its own cursor, so any
/// number of concurrent readers can share one file handle without sharing
/// an offset. (`File::try_clone` dups the fd but SHARES the underlying file
/// offset — seek+read through clones races across threads.)
pub struct CountingRead {
    file: Arc<File>,
    pos: u64,
    counter: ByteCounter,
}

impl Read for CountingRead {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.file.read_at(buf, self.pos)?;
        self.pos += n as u64;
        self.counter.add(n as u64);
        Ok(n)
    }
}

impl ChunkReader for CountingFile {
    type T = CountingRead;

    fn get_read(&self, start: u64) -> Result<Self::T, ParquetError> {
        Ok(CountingRead {
            file: self.file.clone(),
            pos: start,
            counter: self.counter.clone(),
        })
    }

    fn get_bytes(&self, start: u64, length: usize) -> Result<Bytes, ParquetError> {
        let mut buf = vec![0u8; length];
        self.file.read_exact_at(&mut buf, start)?;
        self.counter.add(length as u64);
        Ok(buf.into())
    }
}
