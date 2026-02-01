// include/random.h
#ifndef RL2_RANDOM_H_
#define RL2_RANDOM_H_

#include <cstdint>
namespace arelto {

uint32_t GenerateRandomInt(uint32_t min_val, uint32_t max_val);
float GenerateRandomFloat(float min_val, float max_val);

}  // namespace arelto

#endif
