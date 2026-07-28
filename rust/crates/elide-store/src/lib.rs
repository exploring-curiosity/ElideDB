//! elide-store: the ElideDB store engine core.
//!
//! Reads the language-neutral store contract (Parquet tables + JSON
//! transaction log) that the Python engine writes; every read is counted.

pub mod count;
pub mod log;
pub mod predicate;
pub mod scan;
pub mod store;

pub use count::{ByteCounter, CountingFile};
pub use log::{FileEntry, TableLog, TableState};
pub use predicate::{Op, Predicate, Value};
pub use scan::{scan, scan_where, ScanResult, ScanStats};
pub use store::{Store, StoreMeta, TableSummary};
