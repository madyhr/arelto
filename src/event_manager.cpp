// src/event_manager.cpp
#include "event_manager.h"

namespace arelto {

void EventManager::Emit(GameEvent event) {
  events_.push_back(event);
}

// Dispatches all queued events to their respective subscribers.
// Implementation notes:
// 1. An index-based loop instead of range-for is used so that handlers can safely call Emit()
// during dispatch.
// 2. As handlers may call Emit(), they can cause the `events_` vector to reallocate its memory.
// The value-copy instead of const ref of `typed_event` prevents Use-After-Free bugs by
// decoupling the event from the vector's potentially volatile memory.
// 3. Each event base type has its own list of listener functions to call when that event is heard.
// Therefore, we need to first get the list specific to this event base type and then loop over
// and call all listener functions with this event base type and the event context.
void EventManager::Dispatch(EventContext& event_context) {
  for (size_t i = 0; i < events_.size(); ++i) {
    std::visit(
        [&](auto typed_event) {
          // We want to get the clean base type of the unwrapped event to use it for the handler lookup.
          using T = std::decay_t<decltype(typed_event)>;
          for (auto& [subscription_id, handler] :
               std::get<HandlerList<T>>(handlers_)) {
            handler(typed_event, event_context);
          }
        },
        events_[i]);
  }
}

// Performs an immediate dispatch of the input event to all its subscribers.
// Implementation notes:
// 1. In contrast to `Dispatch` it does not iterate over the `events_` vector, but it only
// dispatches the input event. This means that if `DispatchImmediate` is called with an
// event that emits another event, then this other event is added to `events_` and is not
// immediately dispatched like its parent event.
void EventManager::DispatchImmediate(GameEvent event,
                                     EventContext& event_context) {
  std::visit(
      [&](auto typed_event) {
        // We want to get the clean base type of the unwrapped event to use it for the handler lookup.
        using T = std::decay_t<decltype(typed_event)>;
        for (auto& [subscription_id, handler] :
             std::get<HandlerList<T>>(handlers_)) {
          handler(typed_event, event_context);
        }
      },
      event);
}

void EventManager::Flush() {
  events_.clear();
}

const std::vector<GameEvent>& EventManager::GetEvents() const {
  return events_;
}

}  // namespace arelto
