#include <catch2/catch_test_macros.hpp>

#include "streetdex/metrics/metrics.hpp"

TEST_CASE("metrics scope accumulates and unbinds", "[metrics]") {
  sdx::metrics::IoStats a;
  {
    sdx::metrics::Scope s(a);
    sdx::metrics::count(sdx::metrics::Cat::sdx_data, 100);
    sdx::metrics::count(sdx::metrics::Cat::sdx_footer, 8);
  }
  sdx::metrics::count(sdx::metrics::Cat::sdx_data, 999);  // outside scope
  CHECK(a.total_bytes() == 108);
  CHECK(a.bytes[static_cast<size_t>(sdx::metrics::Cat::sdx_data)] == 100);
  CHECK(a.ops[static_cast<size_t>(sdx::metrics::Cat::sdx_data)] == 1);
}
