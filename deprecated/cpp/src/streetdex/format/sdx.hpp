#pragma once
// SDX v1 — StreetDex columnar sensor format. Byte-exact layout in
// docs/FORMAT.md; every field there has a WHY. Summary of the shape:
//
//   "SDX1" | chunk0(col0 ts, col1, ...) | chunk1 ... | FOOTER | len u32 | "SDX1"
//
// - footer at END: single-pass writer, 2-I/O reader (Parquet's trick)
// - magic at BOTH ends: cheap wrong-file rejection + torn-write detection
// - per-chunk per-column min/max in the footer = zone maps; pruning never
//   touches the data region
// - ts column (col 0) is non-decreasing file-wide: makes chunk pruning a
//   binary search and makes ts zone maps perfectly tight
// - raw fixed-width values, no compression: mmap + pointer cast = zero-copy;
//   v1 takes the extreme random-access position (the Lance/Vortex tension).

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "streetdex/core/error.hpp"
#include "streetdex/core/mmap_file.hpp"
#include "streetdex/core/time.hpp"

namespace sdx {

inline constexpr char kSdxMagic[4] = {'S', 'D', 'X', '1'};
inline constexpr uint16_t kSdxVersion = 1;
inline constexpr uint32_t kDefaultChunkTargetRows = 4096;

enum class ColType : uint8_t {
  TS_NS = 0,  // i64 ns since epoch (canonical timeline); column 0 only
  F64 = 1,
  F32 = 2,
  I64 = 3,
  I16 = 4,  // v1 extension: raw 16-bit PCM stays byte-identical to source
};

inline size_t col_type_width(ColType t) {
  switch (t) {
    case ColType::TS_NS: return 8;
    case ColType::F64: return 8;
    case ColType::F32: return 4;
    case ColType::I64: return 8;
    case ColType::I16: return 2;
  }
  return 0;
}

struct ColumnSpec {
  std::string name;
  ColType type;
};

// Zone map bounds are stored as the raw 8-byte bit pattern of the column type
// (narrow types occupy the low bytes, high bytes zero).
struct ColSpan {
  uint64_t data_offset = 0;
  uint64_t data_length = 0;
  uint64_t min_raw = 0;
  uint64_t max_raw = 0;
};

struct ChunkEntry {
  uint32_t row_count = 0;
  std::vector<ColSpan> cols;  // one per column, same order as schema

  TimeNs ts_min() const { return static_cast<TimeNs>(cols[0].min_raw); }
  TimeNs ts_max() const { return static_cast<TimeNs>(cols[0].max_raw); }
};

struct SdxMeta {
  std::string stream_id;
  std::string units;
  TimeNs clock_offset_ns = 0;  // this stream's local clock minus canonical
  uint32_t chunk_target_rows = kDefaultChunkTargetRows;
  std::vector<ColumnSpec> columns;
  uint64_t row_count = 0;
  TimeNs min_ts = 0;
  TimeNs max_ts = 0;
  uint64_t file_size = 0;    // elision denominator for this file
  uint64_t footer_bytes = 0; // footer + trailing 8B (directory cost)
};

// ---- Writer ----------------------------------------------------------------
class SdxWriter {
 public:
  static Result<SdxWriter> create(const std::string& path,
                                  std::string stream_id, std::string units,
                                  TimeNs clock_offset_ns,
                                  std::vector<ColumnSpec> value_columns,
                                  uint32_t chunk_target_rows =
                                      kDefaultChunkTargetRows);
  ~SdxWriter();
  SdxWriter(SdxWriter&&) noexcept;
  SdxWriter& operator=(SdxWriter&&) = delete;
  SdxWriter(const SdxWriter&) = delete;

  // Append `rows` rows: ts[i] plus, per value column j, rows values of that
  // column's type densely packed at value_cols[j]. Enforces ts monotonicity
  // (the invariant every pruning mechanism depends on).
  Result<void> append(size_t rows, const int64_t* ts,
                      const void* const* value_cols);

  // Flush the tail chunk, write footer + trailing magic. Must be called;
  // destructor without finish() leaves a torn file (detectable by readers).
  Result<void> finish();

 private:
  SdxWriter() = default;
  Result<void> flush_chunk();
  struct Impl;
  Impl* impl_ = nullptr;
};

// ---- Reader ----------------------------------------------------------------
struct ColumnSlice {
  const uint8_t* data = nullptr;  // into the mmap; zero-copy
  size_t rows = 0;
};

struct RangeScan {
  // spans[c][k]: for requested column c, the k-th contiguous run of rows.
  // First/last runs are row-trimmed to [t0, t1]; interior runs are whole
  // chunks (sortedness guarantees they are fully inside the window).
  std::vector<std::vector<ColumnSlice>> spans;
  std::vector<uint32_t> column_ids;  // schema index per requested column
  size_t total_rows = 0;
  uint32_t chunks_scanned = 0;
  uint32_t chunks_total = 0;
};

class SdxReader {
 public:
  // Two logical I/Os: last 8 bytes (magic + footer_length), then the footer.
  // Both are charged to metrics as sdx_footer.
  static Result<SdxReader> open(const std::string& path);

  const SdxMeta& meta() const { return meta_; }

  // Zone-map pruned range scan. `columns` empty => all columns.
  Result<RangeScan> scan(TimeNs t0, TimeNs t1,
                         std::span<const std::string> columns = {}) const;

  // Convenience materializers for the align layer (F64 output).
  static std::vector<int64_t> gather_i64(const RangeScan& s, size_t col_idx);
  static std::vector<double> gather_f64(const RangeScan& s, size_t col_idx,
                                        ColType type);

  const std::vector<ChunkEntry>& chunks() const { return chunks_; }

 private:
  MmapFile file_;
  SdxMeta meta_;
  std::vector<ChunkEntry> chunks_;
};

}  // namespace sdx
