#ifndef RL2_ITEMS_H_
#define RL2_ITEMS_H_

#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "entity.h"
#include "event_manager.h"
#include "scene.h"
#include "types.h"
#include "upgrades.h"

namespace arelto {

class ItemManager;

// Class for items whose effect trigger on events.
class ItemTriggerEffect {
 public:
  virtual ~ItemTriggerEffect() = default;
  ItemTriggerEffect() = default;
  ItemTriggerEffect(const ItemTriggerEffect&) = delete;
  ItemTriggerEffect& operator=(const ItemTriggerEffect&) = delete;
  ItemTriggerEffect(ItemTriggerEffect&&) = default;
  ItemTriggerEffect& operator=(ItemTriggerEffect&&) = default;

  // Child classes override only the events they trigger on. Defaults are no-ops
  // so that adding a new event trigger here does not force every existing item to
  // implement it.
  virtual void OnEnemyKilled(const EnemyKilledEvent&, EventContext&) {}
};

// Heal the player for heal_amount HP each time an enemy is killed.
class HealOnKillEffect : public ItemTriggerEffect {
 public:
  explicit HealOnKillEffect(int heal_amount) : heal_amount_(heal_amount) {}
  void OnEnemyKilled(const EnemyKilledEvent&,
                     EventContext& event_context) override {
    event_context.scene.player.TakeHealing(heal_amount_);
    event_context.event_manager.Emit(PlayerHealedEvent{heal_amount_});
  }

 private:
  int heal_amount_;
};

enum ItemId : int {
  elia_armor_plate = 0,
  damodei_claw,
  volmnih_boots,
  sarto_button_bible,
  count
};

enum class ItemUpgradeType : int {
  armor = 0,
  movement_speed,
  global_damage_modifier,
  global_cooldown_modifier,
  count
};

inline bool IsHigherBetter(ItemUpgradeType type) {
  switch (type) {
    case ItemUpgradeType::armor:
    case ItemUpgradeType::movement_speed:
    case ItemUpgradeType::global_damage_modifier:
      return true;
    case ItemUpgradeType::global_cooldown_modifier:
      return false;
    default:
      return true;
  }
}

struct ItemStatSpec {
  ItemUpgradeType stat_type;
  ModifierType modifier_type;
  float value;
  std::string description;
};

struct ItemTriggerSpec {
  std::string description;
  std::function<std::unique_ptr<ItemTriggerEffect>()> make_effect;
};

struct Item {
  ItemId id;
  std::string name;
  std::vector<ItemStatSpec> stat_specs;
  std::vector<ItemTriggerSpec> trigger_specs;
};

class ItemArchive {
 public:
  ItemArchive() { LoadItems(); }

  const Item& GetItem(ItemId id) {
    size_t index = static_cast<size_t>(id);
    return archive_[index];
  }

 private:
  std::vector<Item> archive_;

  void LoadItems() {
    archive_.resize(ItemId::count);
    archive_[ItemId::elia_armor_plate] = {
        ItemId::elia_armor_plate,
        "Skewer-safe Armorplate of Elia",
        {ItemStatSpec{ItemUpgradeType::armor, ModifierType::flat, 1.0f,
                      "Increase Armor"},
         ItemStatSpec{ItemUpgradeType::movement_speed,
                      ModifierType::percent_mult, -0.05f, "Slow Movement"}},
        {}};
    archive_[ItemId::damodei_claw] = {
        ItemId::damodei_claw,
        "Claw of Damodei",
        {},
        {ItemTriggerSpec{"Heal 5 HP on kill", []() {
                           return std::make_unique<HealOnKillEffect>(5);
                         }}}};
    archive_[ItemId::volmnih_boots] = {
        ItemId::volmnih_boots,
        "Volmnih's Asynchronous Boots",
        {ItemStatSpec{ItemUpgradeType::movement_speed,
                      ModifierType::percent_mult, 0.1f,
                      "Increase Movement Speed"}},
        {}};
    archive_[ItemId::sarto_button_bible] = {
        ItemId::sarto_button_bible,
        "Bible of Sarto Button",
        {ItemStatSpec{ItemUpgradeType::global_damage_modifier,
                      ModifierType::percent_mult, -0.05f,
                      "Decrease the damage of all spells."},
         ItemStatSpec{ItemUpgradeType::global_cooldown_modifier,
                      ModifierType::percent_mult, -0.1f,
                      "Decrease the cooldown of all spells."}},
        {}};
  }
};

// A single stat change to apply to the player on pickup.
// NOTE: `value_range` stores both the pre- and post-pickup values
// so the UI can show a before/after.
struct ItemStatModifier {
  ItemUpgradeType stat_type;
  ModifierType modifier_type;
  float raw_value;
  ValueRange value_range;
  std::string description;
  bool is_higher_better;
};

// A single event-triggered effect along with the text the item card should
// render for it.
struct ItemTriggerModifier {
  std::string description;
  std::unique_ptr<ItemTriggerEffect> effect;
};

// An item-based upgrade that can carry any combination of stat modifiers and
// event-triggered modifiers.
// Implementation notes:
// 1. `ItemUpgrade` 's always go through the two-argument `Apply()` overload
// because they may need to register event-triggered effects with the
// ItemManager. The single-argument override exists only to satisfy the
// Upgrade base class interface and should never be called on an ItemUpgrade.

class ItemUpgrade : public Upgrade {
 public:
  ItemUpgrade(ItemId item_id, std::string name,
              std::vector<ItemStatModifier> stat_modifiers,
              std::vector<ItemTriggerModifier> trigger_modifiers)
      : item_id_(item_id),
        name_(std::move(name)),
        stat_modifiers_(std::move(stat_modifiers)),
        trigger_modifiers_(std::move(trigger_modifiers)) {}

  void Apply(Player&) override {}

  // Applies every stat modifier to the player and transfers ownership of every
  // trigger effect to `item_manager`.
  void Apply(Player& player, ItemManager& item_manager);

  ItemId GetItemID() const { return item_id_; }
  std::string GetName() const override { return name_; }

  // Returns the descriptions and values of the item upgrade's stat modifiers and trigger modifiers.
  // This is used for the UI system to display the upgrade's effects on the item selection card.
  std::vector<UpgradeDisplayRow> GetDisplayRows() const override {
    std::vector<UpgradeDisplayRow> rows;
    rows.reserve(stat_modifiers_.size() + trigger_modifiers_.size());
    for (const ItemStatModifier& stat_modifier : stat_modifiers_) {
      bool value_increased =
          stat_modifier.value_range.updated > stat_modifier.value_range.current;
      bool is_improvement =
          (value_increased && stat_modifier.is_higher_better) ||
          (!value_increased && !stat_modifier.is_higher_better);
      rows.push_back(UpgradeDisplayRow{
          stat_modifier.description,
          FormatValue(stat_modifier.value_range.current),
          FormatValue(stat_modifier.value_range.updated), is_improvement});
    }
    for (const ItemTriggerModifier& trigger_modifier : trigger_modifiers_) {
      rows.push_back(UpgradeDisplayRow{trigger_modifier.description, "", ""});
    }
    return rows;
  }

 private:
  static std::string FormatValue(float value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << value;
    return ss.str();
  }

  ItemId item_id_;
  std::string name_;
  std::vector<ItemStatModifier> stat_modifiers_;
  std::vector<ItemTriggerModifier> trigger_modifiers_;
};

}  // namespace arelto

#endif
