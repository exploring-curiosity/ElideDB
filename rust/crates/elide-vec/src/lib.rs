//! elide-vec: the two-tier vector path.
//!
//! Scan tier: 1-bit sign codes in a mmap'd sidecar (`_codes.v<N>.bin`),
//! Hamming-scanned with popcount — 32x fewer bytes than fp32. Refine tier:
//! cosine rerank of the shortlist. Measured on bench frame_vectors before
//! this was built: shortlist(100) + rerank = recall@10 0.98 vs exact.
//! The sidecar is DERIVED: versioned to the table's data version, never
//! committed to the log (the v1 B+ tree lesson).

pub mod codes;
pub mod npy;
pub mod search;

pub use codes::{artifact_path, build_codes, CodesFile};
pub use search::{build_for, exact_top_k, normalize, self_test, tiered_top_k, VecTable};
