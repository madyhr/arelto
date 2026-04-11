// src/items.cpp
#include "items.h"
#include <utility>
#include "item_manager.h"

namespace arelto {

namespace {

Stat* ResolveStat(Player& player, ItemUpgradeType stat_type) {
  switch (stat_type) {
    case ItemUpgradeType::armor:
      return &player.stats_.armor;
    case ItemUpgradeType::movement_speed:
      return &player.stats_.movement_speed;
    case ItemUpgradeType::count:
      return nullptr;
  }
  return nullptr;
}

}  // namespace

void ItemUpgrade::Apply(Player& player, ItemManager& item_manager) {
  for (const ItemStatModifier& stat_modifier : stat_modifiers_) {
    Stat* stat_to_upgrade = ResolveStat(player, stat_modifier.stat_type);
    if (stat_to_upgrade == nullptr) {
      continue;
    }
    Modifier modifier{stat_modifier.raw_value, stat_modifier.modifier_type,
                      nullptr};
    stat_to_upgrade->AddModifier(modifier);
  }

  for (ItemTriggerModifier& trigger_modifier : trigger_modifiers_) {
    item_manager.AddItem(std::move(trigger_modifier.effect));
  }
  // Trigger modifiers are moved to the item manager, so we clear them out
  // of this item upgrade to avoid dangling pointers.
  trigger_modifiers_.clear();
}

}  // namespace arelto
