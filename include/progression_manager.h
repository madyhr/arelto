#ifndef RL2_PROGRESSION_MANAGER_H_
#define RL2_PROGRESSION_MANAGER_H_

#include <memory>
#include "entity.h"
#include "item_manager.h"
#include "scene.h"
#include "upgrades.h"

namespace arelto {

class ProgressionManager {
 public:
  ProgressionManager();
  ~ProgressionManager();

  bool CheckLevelUp(const Player& player);
  void GenerateLevelUpOptions(Scene& scene);
  void GenerateItemOptions(Scene& scene);
  void ApplyLevelUpUpgrade(Scene& scene, int option_index);
  void ApplyItemUpgrade(Scene& scene, ItemManager& item_manager,
                        int option_index);

 private:
  std::unique_ptr<Upgrade> GenerateRandomSpellUpgrade(const Player& player);
  std::unique_ptr<Upgrade> GenerateRandomItem(const Scene& scene);
  bool ApplyUpgrade(Player& player, UpgradeOptions& upgrade_options,
                    int option_index);
};

}  // namespace arelto

#endif
