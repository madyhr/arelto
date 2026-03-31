#ifndef RL2_ITEMS_H_
#define RL2_ITEMS_H_

#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include "entity.h"
#include "event_manager.h"
#include "types.h"
#include "upgrades.h"

namespace arelto {

// -----------------------------------------------------------------------------
// ItemTriggerEffect — base class for items that react to in-game events
// -----------------------------------------------------------------------------

class ItemTriggerEffect {
 public:
  virtual ~ItemTriggerEffect() {
    for (auto& unsubscribe : unsubscribers_) {
      unsubscribe();
    }
  }
  ItemTriggerEffect() = default;
  ItemTriggerEffect(const ItemTriggerEffect&) = delete;
  ItemTriggerEffect& operator=(const ItemTriggerEffect&) = delete;
  ItemTriggerEffect(ItemTriggerEffect&&) = default;
  ItemTriggerEffect& operator=(ItemTriggerEffect&&) = default;

  // Called once when the item is picked up.
  virtual void RegisterHandlers(EventManager& event_manager) = 0;

 protected:
  // Subscribes handler and captures the unsubscription so the destructor can clean up automatically.
  template <typename T>
  void Track(EventManager& event_manager,
             std::function<void(const T&, EventContext&)> handler) {
    SubscriptionId id = event_manager.Subscribe<T>(std::move(handler));
    unsubscribers_.push_back(
        [&event_manager, id]() { event_manager.Unsubscribe<T>(id); });
  }

 private:
  std::vector<std::function<void()>> unsubscribers_;
};

// Heal the player for heal_amount HP each time an enemy is killed.
class HealOnKillEffect : public ItemTriggerEffect {
 public:
  explicit HealOnKillEffect(int heal_amount) : heal_amount_(heal_amount) {}
  void RegisterHandlers(EventManager& event_manager) override {
    Track<EnemyKilledEvent>(event_manager, [this](const EnemyKilledEvent&,
                                                  EventContext& event_context) {
      event_context.player.stats_.health += heal_amount_;
    });
  }

 private:
  int heal_amount_;
};

// -----------------------------------------------------------------------------

enum ItemId : int { elia_armor_plate = 0, count };
enum class ItemUpgradeType : int { armor = 0, count };

struct Item {
  ItemId id;
  std::string name;
  ItemUpgradeType upgrade_type;
  ModifierType modifier_type;
  float value;
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
        ItemId::elia_armor_plate, "Skewer-safe Armorplate of Elia",
        ItemUpgradeType::armor, ModifierType::flat, 1.0f};
  }
};

class ItemStatUpgrade : public Upgrade {
 public:
  ItemStatUpgrade(ItemId item_id, std::string item_name, ItemUpgradeType type,
                  ModifierType modifier_type, ValueRange value_range)
      : item_id_(item_id),
        item_name_(std::move(item_name)),
        type_(type),
        modifier_type_(modifier_type),
        current_value_(value_range.current),
        updated_value_(value_range.updated) {}

  void Apply(Player& player) override {
    Stat* stat_to_upgrade;

    switch (type_) {
      case arelto::ItemUpgradeType::armor:
        stat_to_upgrade = &player.stats_.armor;
        break;
      default:
        break;
    }

    float modifier_value = updated_value_ - current_value_;

    Modifier mod = Modifier{modifier_value, modifier_type_, nullptr};
    stat_to_upgrade->AddModifier(mod);

    return;
  }

  std::string GetDescription() const override {
    switch (type_) {
      case ItemUpgradeType::armor:
        return "Increase Armor";
      default:
        return "Unknown Upgrade";
    }
  }

  ItemUpgradeType GetType() const { return type_; }
  ModifierType GetModifierType() const { return modifier_type_; }
  ItemId GetItemID() const { return item_id_; }
  std::string GetName() const override { return item_name_; }
  std::string GetOldValueString() const override {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << current_value_;
    return ss.str();
  }
  std::string GetNewValueString() const override {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << updated_value_;
    return ss.str();
  }

 private:
  ItemId item_id_;
  std::string item_name_;
  ItemUpgradeType type_;
  ModifierType modifier_type_;
  float current_value_;
  float updated_value_;
};

}  // namespace arelto

#endif
