// src/progression_manager.cpp
#include "progression_manager.h"
#include <algorithm>
#include "abilities.h"
#include "constants/progression_manager.h"
#include "items.h"
#include "types.h"

namespace arelto {

ProgressionManager::ProgressionManager() {}
ProgressionManager::~ProgressionManager() {}

bool ProgressionManager::CheckLevelUp(const Player& player) {
  return player.stats_.exp_points >=
         player.stats_.exp_points_required.GetValueCeil();
}

void ProgressionManager::GenerateLevelUpOptions(Scene& scene) {
  scene.level_up_options.clear();
  for (int i = 0; i < kNumSpellUpgradeOptions; ++i) {
    scene.level_up_options.push_back(GenerateRandomSpellUpgrade(scene.player));
  }
}

void ProgressionManager::GenerateItemOptions(Scene& scene) {
  scene.item_options.clear();
  for (int i = 0; i < kNumItemOptions; ++i) {
    scene.item_options.push_back(GenerateRandomItem(scene));
  }
}

std::unique_ptr<Upgrade> ProgressionManager::GenerateRandomSpellUpgrade(
    const Player& player) {
  SpellId spell_id = static_cast<SpellId>(std::rand() % kNumPlayerSpells);
  SpellUpgradeType type = static_cast<SpellUpgradeType>(
      std::rand() % static_cast<int>(SpellUpgradeType::count));

  float current_value = 0.0f;
  float new_value = 0.0f;

  std::string spell_name = "Unknown Spell";
  const BaseProjectileSpell* spell = player.GetSpell(spell_id);
  if (spell) {
    spell_name = spell->GetName();
  }

  const SpellStats<kNumPlayerSpells>& stats = player.spell_stats_;

  switch (type) {
    case SpellUpgradeType::damage:
      current_value = static_cast<float>(stats.damage[spell_id]);
      new_value = current_value + kDamageUpgradeValue;
      break;
    case SpellUpgradeType::speed:
      current_value = stats.speed[spell_id];
      new_value = current_value + kSpeedUpgradeValue;
      break;
    case SpellUpgradeType::cooldown:
      current_value = stats.cooldown[spell_id];
      // We use the max of (0.1, new_value) to ensure that ability cooldowns
      // are always positive.
      new_value = std::max(0.1f, current_value - kCooldownUpgradeValue);
      break;
    case SpellUpgradeType::size:
      current_value = static_cast<float>(stats.sprite_size[spell_id].width);
      new_value = current_value * kSizeUpgradeFactor;
      break;
    case SpellUpgradeType::count:
      break;
  }

  return std::make_unique<SpellStatUpgrade>(
      spell_id, spell_name, type, ValueRange{current_value, new_value});
}

std::unique_ptr<Upgrade> ProgressionManager::GenerateRandomItem(
    const Scene& scene) {
  ItemId item_id = static_cast<ItemId>(std::rand() % ItemId::count);
  Item item = scene.item_archive->GetItem(item_id);

  float current_value = 0.0f;
  switch (item.upgrade_type) {
    case ItemUpgradeType::armor:
      current_value = scene.player.stats_.armor.GetValue();
      break;
    default:
      break;
  }

  float updated_value = current_value + item.value;

  return std::make_unique<ItemStatUpgrade>(
      item_id, item.name, item.upgrade_type, item.modifier_type,
      ValueRange{current_value, updated_value});
}

void ProgressionManager::ApplyLevelUpUpgrade(Scene& scene, int option_index) {
  bool upgrade =
      ApplyUpgrade(scene.player, scene.level_up_options, option_index);
  if (!upgrade) {
    return;
  }

  scene.player.stats_.level++;
  scene.player.stats_.exp_points -=
      scene.player.stats_.exp_points_required.GetValueCeil();
  scene.player.stats_.exp_points_required.SetBaseValue(
      scene.player.stats_.exp_points_required.GetValue() *
      kPlayerExpRequiredScale);
}

bool ProgressionManager::ApplyUpgrade(Player& player,
                                      UpgradeOptions& upgrade_options,
                                      int option_index) {
  if (option_index < 0 ||
      static_cast<size_t>(option_index) >= upgrade_options.size()) {
    return false;
  }

  const auto& upgrade = upgrade_options[option_index];
  if (upgrade) {
    upgrade->Apply(player);
  }

  return true;
}

void ProgressionManager::ApplyItemUpgrade(Scene& scene, int option_index) {
  bool upgrade = ApplyUpgrade(scene.player, scene.item_options, option_index);
  if (!upgrade) {
    return;
  }
}

}  // namespace arelto
