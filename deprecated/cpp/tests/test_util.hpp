#pragma once
#include <filesystem>
#include <random>
#include <string>

// Per-test scratch dir under the system temp root; removed on destruction.
struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const std::string& tag) {
    std::mt19937_64 rng(std::random_device{}());
    path = std::filesystem::temp_directory_path() /
           ("sdxtest_" + tag + "_" + std::to_string(rng()));
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  std::string str(const std::string& name = "") const {
    return name.empty() ? path.string() : (path / name).string();
  }
};
