#ifndef RL2_CONFIG_PROGRESSION_CONFIG_H_
#define RL2_CONFIG_PROGRESSION_CONFIG_H_

#include <array>
#include "types.h"

namespace arelto {

inline std::array<float, Rarity::Count> MakeDefaultSpellUpgradeRarityWeights() {
  std::array<float, Rarity::Count> weights{};
  weights[Rarity::common] = 16.0f;
  weights[Rarity::rare] = 8.0f;
  weights[Rarity::epic] = 2.0f;
  weights[Rarity::legendary] = 0.5f;
  return weights;
}

struct SpellUpgradeConfig {
  std::array<float, Rarity::Count> rarity_weights =
      MakeDefaultSpellUpgradeRarityWeights();
};

struct ProgressionConfig {
  SpellUpgradeConfig spell_upgrade;
};

ProgressionConfig MakeDefaultProgressionConfig();

}  // namespace arelto

#endif
