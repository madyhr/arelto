#ifndef RL2_PROGRESSION_MANAGER_H_
#define RL2_PROGRESSION_MANAGER_H_

#include <memory>
#include "entity.h"
#include "scene.h"
#include "upgrades.h"

namespace rl2 {

class ProgressionManager {
 public:
  ProgressionManager();
  ~ProgressionManager();

  bool CheckLevelUp(const Player& player);
  void GenerateLevelUpOptions(Scene& scene);
  void ApplyUpgrade(Scene& scene, int option_index);

 private:
  std::unique_ptr<Upgrade> GenerateRandomOption(const Player& player);
};

}  // namespace rl2

#endif
