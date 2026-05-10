// src/progression_manager.cpp
#include "progression_manager.h"
#include <algorithm>
#include "abilities.h"
#include "constants/progression_manager.h"
#include "event_manager.h"
#include "item_manager.h"
#include "items.h"
#include "types.h"

namespace arelto {

ProgressionManager::ProgressionManager() {}
ProgressionManager::~ProgressionManager() {}

void ProgressionManager::Initialize(EventManager& event_manager) {
  event_manager_ = &event_manager;
}

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
  size_t spell_count = player.spell_stats_.damage.size();
  if (spell_count == 0) {
    return nullptr;
  }

  SpellId spell_id = static_cast<SpellId>(std::rand() % spell_count);
  SpellUpgradeType type = static_cast<SpellUpgradeType>(
      std::rand() % static_cast<int>(SpellUpgradeType::count));

  float current_value = 0.0f;
  float new_value = 0.0f;

  std::string spell_name = "Unknown Spell";
  Size2D sprite_size = {};
  const BaseProjectileSpell* spell = player.GetSpell(spell_id);
  if (spell) {
    spell_name = spell->GetName();
    sprite_size = spell->GetSpriteCellSize();
  }

  const SpellStats& stats = player.spell_stats_;

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
      spell_id, spell_name, type, ValueRange{current_value, new_value},
      sprite_size);
}

std::unique_ptr<Upgrade> ProgressionManager::GenerateRandomItem(
    const Scene& scene) {
  ItemId item_id = static_cast<ItemId>(std::rand() % ItemId::count);
  const Item& item = scene.item_archive->GetItem(item_id);

  std::vector<ItemStatModifier> stat_modifiers;
  stat_modifiers.reserve(item.stat_specs.size());
  for (const ItemStatSpec& stat_spec : item.stat_specs) {
    const Stat* stat = ResolveItemStat(scene.player, stat_spec.stat_type);
    if (stat == nullptr) {
      continue;
    }
    float current_value = stat->GetValue();
    float updated_value =
        stat->GetModifiedValue(stat_spec.value, stat_spec.modifier_type);
    stat_modifiers.push_back(ItemStatModifier{
        stat_spec.stat_type, stat_spec.modifier_type, stat_spec.value,
        ValueRange{current_value, updated_value}, stat_spec.description,
        IsHigherBetter(stat_spec.stat_type)});
  }

  std::vector<ItemTriggerModifier> trigger_modifiers;
  trigger_modifiers.reserve(item.trigger_specs.size());
  for (const ItemTriggerSpec& trigger_spec : item.trigger_specs) {
    trigger_modifiers.push_back(ItemTriggerModifier{
        trigger_spec.description, trigger_spec.make_effect()});
  }

  return std::make_unique<ItemUpgrade>(item_id, item.name,
                                       std::move(stat_modifiers),
                                       std::move(trigger_modifiers));
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
      scene.player.exp_required_scale_);
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

void ProgressionManager::ApplyItemUpgrade(Scene& scene,
                                          ItemManager& item_manager,
                                          int option_index) {
  if (option_index < 0 ||
      static_cast<size_t>(option_index) >= scene.item_options.size()) {
    return;
  }

  ItemUpgrade& item_upgrade =
      static_cast<ItemUpgrade&>(*scene.item_options[option_index]);
  item_upgrade.Apply(scene.player, item_manager);
  EventContext event_context{*event_manager_, scene};
  event_manager_->DispatchImmediate(PlayerStatChangeEvent{}, event_context);
  scene.player.AddToInventory(item_upgrade.GetItemID());
}

}  // namespace arelto
