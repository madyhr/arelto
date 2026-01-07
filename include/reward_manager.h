// include/reward_manager.h
#ifndef RL2_REWARD_MANAGER_H_
#define RL2_REWARD_MANAGER_H_

#include <array>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "constants/enemy.h"
#include "scene.h"

namespace rl2 {

using RewardFunction =
    std::function<std::array<float, kNumEnemies>(const Scene& current_scene)>;

struct RewardTerm {

  std::string name;
  float weight;
  RewardFunction func;

  std::array<float, kNumEnemies> last_value;
  // signals are unweighted values
  std::array<float, kNumEnemies> last_signal;
  // used to store accumulated signals
  std::array<float, kNumEnemies> accumulated_signal;
};

class RewardManager {

 public:
  bool Initialize();
  void ResetRewardTerms();
  void UpdateRewardTerms(const Scene& scene);
  void AddTerm(std::string name, float weight, RewardFunction func);
  void RegisterRewardTerms();
  std::array<float, kNumEnemies> CalculateTotalReward(const Scene& scene);
  std::map<std::string, std::array<float, kNumEnemies>> GetLastRewardDict();
  std::map<std::string, std::array<float, kNumEnemies>> GetLastSignalDict();
  void FillRewardBuffer(float* buffer_ptr, int buffer_size, const Scene& scene);
  int GetRewardSize();

 private:
  std::vector<RewardTerm> terms_;
};

}  // namespace rl2

#endif
