// include/random.h
#ifndef RL2_RANDOM_H_
#define RL2_RANDOM_H_

#include <random>
namespace arelto {

static std::mt19937 s_generator(std::random_device{}());
int GenerateRandomInt(int min_val, int max_val);
float GenerateRandomFloat(float min_val, float max_val);

}  // namespace arelto

#endif
