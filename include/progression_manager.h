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
  void ApplyLevelUpUpgrade(Scene& scene, int option_index);

 private:
  std::unique_ptr<Upgrade> GenerateRandomSpellUpgrade(const Player& player);
  bool ApplyUpgrade(Player& player, UpgradeOptions& upgrade_options,
                    int option_index);
};

}  // namespace arelto

#endif
