// src/random.cpp
#include "random.h"
#include <cstdint>
#include <random>

namespace rl2 {

static std::mt19937 s_generator(std::random_device{}());

uint32_t generate_random_int(uint32_t min_val, uint32_t max_val) {
  std::uniform_int_distribution<int> distrib(min_val, max_val);
  return distrib(s_generator);
};

}  // namespace rl2
