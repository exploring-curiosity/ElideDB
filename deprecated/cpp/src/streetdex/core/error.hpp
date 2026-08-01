#pragma once
// Error handling: expected<T, Error> everywhere; exceptions only at the CLI
// boundary. WHY: storage-engine code paths must make failure explicit and
// cheap — a missing chunk is control flow, not a stack unwind.

#include <string>
#include <tl/expected.hpp>

namespace sdx {

enum class Errc {
  io,            // open/read/mmap/stat failed
  bad_format,    // magic/version/layout violation
  bad_argument,  // caller error (t0 > t1, unknown stream, ...)
  not_found,     // file/stream/snapshot missing
  decode,        // libav failure
  internal,      // invariant broken — a bug, not an input problem
};

struct Error {
  Errc code;
  std::string message;
};

template <typename T>
using Result = tl::expected<T, Error>;

inline tl::unexpected<Error> fail(Errc c, std::string msg) {
  return tl::unexpected<Error>(Error{c, std::move(msg)});
}

}  // namespace sdx
