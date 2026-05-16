// src/items.cpp
#include "items.h"
#include <utility>
#include "config/config_manager.h"
#include "config/item_config.h"
#include "config/item_config_yaml.h"  // IWYU pragma: keep
#include "item_manager.h"

namespace arelto {

const std::unordered_map<ItemId, std::string, ItemIdHash> ItemFlavorText{
    {ItemId::elia_armor_plate,
     "Never mistake blind acceleration for progress."},
    {ItemId::damodei_claw,
     "Most of your kind may die out, but that is a sacrifice I am willing to "
     "make."},
    {ItemId::volmnih_boots, "One step at a time? How cute."},
    {ItemId::sarto_button_bible,
     "The 'Bible' began as an introduction to the Karmov schools of magic, but "
     "to many, it became scripture."},
    {ItemId::aiayn_scale, "Pay attention now - this is no ordinary scale."}};

ItemArchive::ItemArchive(const std::filesystem::path& item_config_path) {
  ItemConfig item_config = MakeDefaultItemConfig();
  ConfigManager config_manager;
  config_manager.LoadConfigSectionOrDefault("items", item_config_path,
                                            item_config);
  LoadItems(item_config);
}

const Item& ItemArchive::GetItem(ItemId id) const {
  size_t index = static_cast<size_t>(id);
  return archive_[index];
}

void ItemArchive::LoadItems(const ItemConfig& item_config) {
  archive_.resize(to_index(ItemId::count));
  archive_[to_index(ItemId::elia_armor_plate)] = {
      ItemId::elia_armor_plate,
      "Skewer-safe Armorplate of Elia",
      {ItemStatSpec{ItemUpgradeType::armor, ModifierType::flat, 1.0f,
                    "Increase Armor"},
       ItemStatSpec{ItemUpgradeType::movement_speed, ModifierType::percent_mult,
                    -0.05f, "Slow Movement"}},
      {},
      item_config.items[to_index(ItemId::elia_armor_plate)].flavor_text};
  archive_[to_index(ItemId::damodei_claw)] = {
      ItemId::damodei_claw,
      "Claw of Damodei",
      {},
      {ItemTriggerSpec{"Heal 5 HP on kill",
                       []() { return std::make_unique<HealOnKillEffect>(5); }}},
      item_config.items[to_index(ItemId::damodei_claw)].flavor_text};
  archive_[to_index(ItemId::volmnih_boots)] = {
      ItemId::volmnih_boots,
      "Volmnih's Asynchronous Boots",
      {ItemStatSpec{ItemUpgradeType::movement_speed, ModifierType::percent_mult,
                    0.1f, "Increase Movement Speed"}},
      {},
      item_config.items[to_index(ItemId::volmnih_boots)].flavor_text};
  archive_[to_index(ItemId::sarto_button_bible)] = {
      ItemId::sarto_button_bible,
      "Bible of Sarto Button",
      {ItemStatSpec{ItemUpgradeType::global_damage_modifier,
                    ModifierType::percent_mult, -0.05f,
                    "Decrease the damage of all spells."},
       ItemStatSpec{ItemUpgradeType::global_cooldown_modifier,
                    ModifierType::percent_mult, -0.1f,
                    "Decrease the cooldown of all spells."}},
      {},
      item_config.items[to_index(ItemId::sarto_button_bible)].flavor_text};
  archive_[to_index(ItemId::aiayn_scale)] = {
      ItemId::aiayn_scale,
      "Aiayn's Ever- Transforming Scale",
      {ItemStatSpec{ItemUpgradeType::max_health, ModifierType::flat, 50.0f,
                    "Increase Max Health Points"},
       ItemStatSpec{ItemUpgradeType::size, ModifierType::percent_mult, 0.05f,
                    "Increase Player Size"}},
      {},
      item_config.items[to_index(ItemId::aiayn_scale)].flavor_text};
}

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
