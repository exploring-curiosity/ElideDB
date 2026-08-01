// SDX golden/round-trip tests: every FORMAT.md §1.6 invariant becomes a check.
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>

#include "streetdex/format/sdx.hpp"
#include "streetdex/metrics/metrics.hpp"
#include "test_util.hpp"

using namespace sdx;

namespace {

std::string write_basic(const TempDir& td, uint32_t chunk_rows = 4) {
  const std::string path = td.str("basic.sdx");
  auto w = SdxWriter::create(path, "imu0", "m/s^2", 12345,
                             {{"accel", ColType::F64},
                              {"gyro", ColType::F32},
                              {"count", ColType::I64},
                              {"pcm", ColType::I16}},
                             chunk_rows);
  REQUIRE(w.has_value());
  // 10 rows over ts 100..1000 -> 3 chunks of 4/4/2 at chunk_rows=4
  std::vector<int64_t> ts, count;
  std::vector<double> accel;
  std::vector<float> gyro;
  std::vector<int16_t> pcm;
  for (int i = 0; i < 10; ++i) {
    ts.push_back(100 * (i + 1));
    accel.push_back(1.5 * i);
    gyro.push_back(-0.5f * i);
    count.push_back(1000 + i);
    pcm.push_back(static_cast<int16_t>(-100 * i));
  }
  const void* cols[] = {accel.data(), gyro.data(), count.data(), pcm.data()};
  REQUIRE(w->append(10, ts.data(), cols).has_value());
  REQUIRE(w->finish().has_value());
  return path;
}

}  // namespace

TEST_CASE("SDX round-trip preserves schema, meta and values", "[sdx]") {
  TempDir td("sdxrt");
  const auto path = write_basic(td);
  auto r = SdxReader::open(path);
  REQUIRE(r.has_value());
  const auto& m = r->meta();
  CHECK(m.stream_id == "imu0");
  CHECK(m.units == "m/s^2");
  CHECK(m.clock_offset_ns == 12345);
  CHECK(m.row_count == 10);
  CHECK(m.min_ts == 100);
  CHECK(m.max_ts == 1000);
  REQUIRE(m.columns.size() == 5);
  CHECK(m.columns[0].name == "ts");
  CHECK(m.columns[4].type == ColType::I16);
  CHECK(r->chunks().size() == 3);  // 4 + 4 + 2

  auto scan = r->scan(0, 2000);
  REQUIRE(scan.has_value());
  CHECK(scan->total_rows == 10);
  auto ts = SdxReader::gather_i64(*scan, 0);
  auto accel = SdxReader::gather_f64(*scan, 1, ColType::F64);
  auto pcm = SdxReader::gather_f64(*scan, 4, ColType::I16);
  CHECK(ts.front() == 100);
  CHECK(ts.back() == 1000);
  CHECK(accel[3] == 4.5);
  CHECK(pcm[2] == -200.0);
}

TEST_CASE("SDX zone maps prune non-overlapping chunks", "[sdx]") {
  TempDir td("sdxprune");
  const auto path = write_basic(td);  // chunks cover [100,400][500,800][900,1000]
  auto r = SdxReader::open(path);
  REQUIRE(r.has_value());

  metrics::IoStats io;
  {
    metrics::Scope s(io);
    auto scan = r->scan(500, 700);  // only chunk 1
    REQUIRE(scan.has_value());
    CHECK(scan->chunks_scanned == 1);
    CHECK(scan->chunks_total == 3);
    CHECK(scan->total_rows == 3);  // ts 500,600,700
    auto ts = SdxReader::gather_i64(*scan, 0);
    CHECK(ts == std::vector<int64_t>{500, 600, 700});
  }
  // Charged: exactly chunk 1's column spans (4 rows: 8+8+4+8+2 = 30 B/row,
  // padded per column — but never chunk 0 or 2).
  const auto data =
      io.bytes[static_cast<size_t>(metrics::Cat::sdx_data)];
  CHECK(data == 4 * (8 + 8 + 4 + 8 + 2));

  // Empty window entirely off the end: zero data bytes.
  metrics::IoStats io2;
  {
    metrics::Scope s(io2);
    auto scan = r->scan(5000, 6000);
    REQUIRE(scan.has_value());
    CHECK(scan->total_rows == 0);
  }
  CHECK(io2.bytes[static_cast<size_t>(metrics::Cat::sdx_data)] == 0);
}

TEST_CASE("SDX column projection reads only requested columns", "[sdx]") {
  TempDir td("sdxproj");
  const auto path = write_basic(td);
  auto r = SdxReader::open(path);
  REQUIRE(r.has_value());
  metrics::IoStats io;
  {
    metrics::Scope s(io);
    const std::string want[] = {"accel"};
    auto scan = r->scan(500, 700, want);
    REQUIRE(scan.has_value());
    REQUIRE(scan->spans.size() == 1);
    auto accel = SdxReader::gather_f64(*scan, 0, ColType::F64);
    CHECK(accel.size() == 3);
  }
  // accel span (4*8) + ts span for row-trim (4*8), nothing else
  CHECK(io.bytes[static_cast<size_t>(metrics::Cat::sdx_data)] == 32 + 32);
}

TEST_CASE("SDX rejects out-of-order timestamps at write time", "[sdx]") {
  TempDir td("sdxmono");
  auto w = SdxWriter::create(td.str("bad.sdx"), "s", "", 0,
                             {{"v", ColType::F64}});
  REQUIRE(w.has_value());
  std::vector<int64_t> ts = {10, 5};
  std::vector<double> v = {1, 2};
  const void* cols[] = {v.data()};
  auto res = w->append(2, ts.data(), cols);
  REQUIRE_FALSE(res.has_value());
  CHECK(res.error().code == Errc::bad_argument);
}

TEST_CASE("SDX detects torn files (trailing magic gone)", "[sdx]") {
  TempDir td("sdxtorn");
  const auto path = write_basic(td);
  // Truncate the last 3 bytes: trailing magic destroyed.
  std::error_code ec;
  const auto sz = std::filesystem::file_size(path, ec);
  std::filesystem::resize_file(path, sz - 3, ec);
  auto r = SdxReader::open(path);
  REQUIRE_FALSE(r.has_value());
  CHECK(r.error().code == Errc::bad_format);
}

TEST_CASE("SDX rejects unknown columns and t0>t1", "[sdx]") {
  TempDir td("sdxargs");
  const auto path = write_basic(td);
  auto r = SdxReader::open(path);
  REQUIRE(r.has_value());
  const std::string bad[] = {"nope"};
  CHECK_FALSE(r->scan(0, 10, bad).has_value());
  CHECK_FALSE(r->scan(10, 0).has_value());
}
