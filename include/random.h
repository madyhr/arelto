// include/random.h
#ifndef RL2_RANDOM_H_
#define RL2_RANDOM_H_

#include <vector>
namespace arelto {

int GenerateRandomInt(int min_val, int max_val);
float GenerateRandomFloat(float min_val, float max_val);
int SampleFromDiscreteDist(std::vector<float> weights);

}  // namespace arelto

#endif
