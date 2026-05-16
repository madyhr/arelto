// src/items.cpp
#include "items.h"
#include <utility>
#include "item_manager.h"

namespace arelto {

const std::unordered_map<ItemId, std::string, ItemIdHash> ItemFlavorText{
    {ItemId::elia_armor_plate,
     "The moment the world starts moving fast around you is exactly the moment "
     "you should consider slowing down and taking safe, small steps."},
    {ItemId::damodei_claw,
     "Most of your kind may die out, but that is a sacrifice I am willing to "
     "make."},
    {ItemId::volmnih_boots, "One step at a time? How cute."},
    {ItemId::sarto_button_bible,
     "The 'Bible' started as an entry-point to the mastery of the Karmov "
     "schools of magic, but to many it became so much more."},
    {ItemId::aiayn_scale, "Pay attention now - this is no ordinary scale."}};

const Stat* ResolveItemStat(const Player& player, ItemUpgradeType stat_type) {
  switch (stat_type) {
    case ItemUpgradeType::armor:
      return &player.stats_.armor;
    case ItemUpgradeType::movement_speed:
      return &player.stats_.movement_speed;
    case ItemUpgradeType::max_health:
      return &player.stats_.max_health;
    case ItemUpgradeType::size:
      // As player size is defined used class `StatSize`, the size is defined
      // by only its `width_` property with class `Stat`.
      return &player.stats_.size.width_;
    case ItemUpgradeType::global_damage_modifier:
      return &player.stats_.global_damage_modifier;
    case ItemUpgradeType::global_cooldown_modifier:
      return &player.stats_.global_cooldown_modifier;
    case ItemUpgradeType::count:
      return nullptr;
  }
  return nullptr;
}

Stat* ResolveItemStat(Player& player, ItemUpgradeType stat_type) {
  return const_cast<Stat*>(
      ResolveItemStat(static_cast<const Player&>(player), stat_type));
}

void ItemUpgrade::Apply(Player& player, ItemManager& item_manager) {
  for (const ItemStatModifier& stat_modifier : stat_modifiers_) {
    Stat* stat_to_upgrade = ResolveItemStat(player, stat_modifier.stat_type);
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
