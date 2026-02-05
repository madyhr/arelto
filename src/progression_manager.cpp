// src/progression_manager.cpp
#include "progression_manager.h"
#include <algorithm>
#include "abilities.h"
#include "constants/progression_manager.h"
#include "types.h"

namespace arelto {

ProgressionManager::ProgressionManager() {}
ProgressionManager::~ProgressionManager() {}

bool ProgressionManager::CheckLevelUp(const Player& player) {
  return player.stats_.exp_points >= player.stats_.exp_points_required;
}

int ProgressionManager::ApplyExpScalingLaw(const int& current_exp_req) {
  return static_cast<int>(current_exp_req * kPlayerExpRequiredScale);
};

void ProgressionManager::GenerateLevelUpOptions(Scene& scene) {
  scene.level_up_options.clear();
  for (int i = 0; i < kNumUpgradeOptions; ++i) {
    scene.level_up_options.push_back(GenerateRandomOption(scene.player));
  }
}

std::unique_ptr<Upgrade> ProgressionManager::GenerateRandomOption(
    const Player& player) {
  SpellId spell_id = static_cast<SpellId>(std::rand() % kNumPlayerSpells);
  UpgradeType type = static_cast<UpgradeType>(
      std::rand() % static_cast<int>(UpgradeType::count));

  float current_value = 0.0f;
  float new_value = 0.0f;

  std::string spell_name = "Unknown Spell";
  const BaseProjectileSpell* spell = player.GetSpell(spell_id);
  if (spell) {
    spell_name = spell->GetName();
  }

  const SpellStats<kNumPlayerSpells>& stats = player.spell_stats_;

  switch (type) {
    case UpgradeType::damage:
      current_value = static_cast<float>(stats.damage[spell_id]);
      new_value = current_value + kDamageUpgradeValue;
      break;
    case UpgradeType::speed:
      current_value = stats.speed[spell_id];
      new_value = current_value + kSpeedUpgradeValue;
      break;
    case UpgradeType::cooldown:
      current_value = stats.cooldown[spell_id];
      // We use the max of (0.1, new_value) to ensure that ability cooldowns
      // are always positive.
      new_value = std::max(0.1f, current_value - kCooldownUpgradeValue);
      break;
    case UpgradeType::size:
      current_value = static_cast<float>(stats.sprite_size[spell_id].width);
      new_value = current_value * kSizeUpgradeFactor;
      break;
    case UpgradeType::count:
      break;
  }

  return std::make_unique<SpellStatUpgrade>(spell_id, spell_name, type,
                                            current_value, new_value);
}

void ProgressionManager::ApplyUpgrade(Scene& scene, int option_index) {
  if (option_index < 0 ||
      static_cast<size_t>(option_index) >= scene.level_up_options.size()) {
    return;
  }

  const auto& upgrade = scene.level_up_options[option_index];
  if (upgrade) {
    upgrade->Apply(scene.player);
  }

  scene.player.stats_.level++;
  scene.player.stats_.exp_points -= scene.player.stats_.exp_points_required;
  scene.player.stats_.exp_points_required = static_cast<int>(
      scene.player.stats_.exp_points_required * kPlayerExpRequiredScale);
}

}  // namespace arelto
