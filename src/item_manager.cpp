// src/item_manager.cpp
#include "item_manager.h"
#include <utility>
#include "event_manager.h"

namespace arelto {

void ItemManager::Initialize(EventManager& event_manager) {
  event_manager.Subscribe<EnemyKilledEvent>(
      [this](const EnemyKilledEvent& event, EventContext& context) {
        for (const auto& item : items_) {
          item->OnEnemyKilled(event, context);
        }
      });
}

void ItemManager::AddItem(std::unique_ptr<ItemTriggerEffect> item) {
  items_.push_back(std::move(item));
}

void ItemManager::RemoveAllItems() {
  items_.clear();
}

}  // namespace arelto
