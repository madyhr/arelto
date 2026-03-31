// src/random.cpp
#include "random.h"
#include <random>

namespace arelto {

static std::mt19937 s_generator(std::random_device{}());

int GenerateRandomInt(int min_val, int max_val) {
  std::uniform_int_distribution<int> distrib(min_val, max_val);
  return distrib(s_generator);
};

float GenerateRandomFloat(float min_val, float max_val) {
  std::uniform_real_distribution<float> distrib(min_val, max_val);
  return distrib(s_generator);
};

}  // namespace arelto
