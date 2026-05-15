#ifndef RL2_CONFIG_PROGRESSION_CONFIG_H_
#define RL2_CONFIG_PROGRESSION_CONFIG_H_

#include <array>
#include "types.h"

namespace arelto {

inline std::array<float, to_index(Rarity::count)>
MakeDefaultSpellUpgradeRarityWeights() {
  std::array<float, to_index(Rarity::count)> weights{};
  weights[to_index(Rarity::common)] = 16.0f;
  weights[to_index(Rarity::rare)] = 8.0f;
  weights[to_index(Rarity::epic)] = 2.0f;
  weights[to_index(Rarity::legendary)] = 0.5f;
  return weights;
}

struct SpellUpgradeConfig {
  std::array<float, to_index(Rarity::count)> rarity_weights =
      MakeDefaultSpellUpgradeRarityWeights();
};

struct ProgressionConfig {
  SpellUpgradeConfig spell_upgrade;
};

ProgressionConfig MakeDefaultProgressionConfig();

}  // namespace arelto

#endif
