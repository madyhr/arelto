// include/item_manager.h
#ifndef RL2_ITEM_MANAGER_H_
#define RL2_ITEM_MANAGER_H_

#include <memory>
#include <vector>
#include "items.h"

namespace arelto {

class EventManager;

// Owns every item picked up during a run and is the single subscriber on the
// EventManager for item-reactive events.
class ItemManager {
 public:
  // Called exactly once at game startup. Registers one handler per
  // item-relevant event type that forwards the typed event to every owned
  // item.
  // NOTE: This could be refactored to be more data-driven (e.g. by having items register their own
  // handlers), but this is simpler to implement given the small number of items and events currently present.
  void Initialize(EventManager& event_manager);

  // Transfers ownership of the item to this manager and adds it to the `items_` vector.
  void AddItem(std::unique_ptr<ItemTriggerEffect> item);

  void RemoveAllItems();

 private:
  std::vector<std::unique_ptr<ItemTriggerEffect>> items_;
};

}  // namespace arelto

#endif
