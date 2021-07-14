#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace mica::experiments::models {

std::vector<float> ReadAll(const std::string &file, const std::string &dir);

std::vector<float> RandomData(size_t size);
} // namespace mica::experiments::models