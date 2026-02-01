#ifndef RL2_PROGRESSION_MANAGER_H_
#define RL2_PROGRESSION_MANAGER_H_

#include <memory>
#include "entity.h"
#include "scene.h"
#include "upgrades.h"

namespace arelto {

class ProgressionManager {
 public:
  ProgressionManager();
  ~ProgressionManager();

  bool CheckLevelUp(const Player& player);
  void GenerateLevelUpOptions(Scene& scene);
  void ApplyUpgrade(Scene& scene, int option_index);
  int ApplyExpScalingLaw(const int& current_exp_req);

 private:
  std::unique_ptr<Upgrade> GenerateRandomOption(const Player& player);
};

}  // namespace arelto

#endif
