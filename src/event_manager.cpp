// src/event_manager.cpp
#include "event_manager.h"

namespace arelto {

void EventManager::Emit(GameEvent event) {
  events_.push_back(event);
}

void EventManager::Flush() {
  events_.clear();
}

const std::vector<GameEvent>& EventManager::GetEvents() const {
  return events_;
}

std::vector<GameEvent> EventManager::GetEvents(EventType type) const {
  std::vector<GameEvent> filtered;
  for (const auto& event : events_) {
    if (event.type == type) {
      filtered.push_back(event);
    }
  }
  return filtered;
}

}  // namespace arelto
