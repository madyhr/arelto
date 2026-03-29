// src/event_manager.cpp
#include "event_manager.h"

namespace arelto {

void EventManager::Emit(GameEvent event) {
  events_.push_back(event);
}

void EventManager::Dispatch(EventContext& event_context) {
  // We are using an index-based loop instead of range-for so that handlers can safely call
  // Emit() during dispatch.
  for (size_t i = 0; i < events_.size(); ++i) {
    std::visit(
        [&](const auto& typed_event) {
          // We want to get the clean base type of the unwrapped event to use it for the handler lookup.
          using T = std::decay_t<decltype(typed_event)>;
          // Each event base type has its own list of listener functions to call when that event is heard.
          // Therefore, we need to first get the list specific to this event base type and then loop over
          // and call all listener functions with this event base type and the event context.
          for (auto& [subscription_id, handler] : std::get<HandlerList<T>>(handlers_)) {
            handler(typed_event, event_context);
          }
        },
        events_[i]);
  }
}

void EventManager::Flush() {
  events_.clear();
}

const std::vector<GameEvent>& EventManager::GetEvents() const {
  return events_;
}

}  // namespace arelto
