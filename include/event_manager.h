// include/event_manager.h
#ifndef RL2_EVENT_MANAGER_H_
#define RL2_EVENT_MANAGER_H_

#include <algorithm>
#include <functional>
#include <tuple>
#include <variant>
#include <vector>

namespace arelto {

class Player;  // forward declaration for EventContext

using SubscriptionId = int;

// -----------------------------------------------------------------------------
// Typed event structs
// -----------------------------------------------------------------------------

struct EnemyKilledEvent {
  int enemy_idx;
  int proj_idx;
  int damage;
};
struct PlayerDamagedEvent {
  int enemy_idx;
  int damage_dealt;
};
struct GemCollectedEvent {
  int gem_idx;
  int exp_value;
};
struct ProjectileDestroyedEvent {
  int proj_idx;
};
struct ChestOpenedEvent {
  int chest_idx;
};
struct PlayerDeadEvent {};

using GameEvent =
    std::variant<EnemyKilledEvent, PlayerDamagedEvent, GemCollectedEvent,
                 ProjectileDestroyedEvent, ChestOpenedEvent, PlayerDeadEvent>;

// The context that is passed to every handler during a Dispatch call.
struct EventContext {
  Player& player;
};

// -----------------------------------------------------------------------------
// EventManager
// -----------------------------------------------------------------------------

class EventManager {
 public:
  void Emit(GameEvent event);

  // Adds an event handler to a specific event base type and returns a SubscriptionId
  // that can be passed to Unsubscribe<T> to remove the handler.
  template <typename T>
  SubscriptionId Subscribe(
      std::function<void(const T&, EventContext&)> handler) {
    SubscriptionId id = next_subscription_id_++;
    std::get<HandlerList<T>>(handlers_).emplace_back(id, std::move(handler));
    return id;
  }

  // Removes the handler registered under the given id for event type T.
  // NOTE: Calling Unsubscribe from within a handler during Dispatch is not supported.
  template <typename T>
  void Unsubscribe(SubscriptionId id) {
    auto& handler_list = std::get<HandlerList<T>>(handlers_);
    auto handler_to_unsubscribe_from = std::find_if(
        handler_list.begin(), handler_list.end(),
        [id](const HandlerEntry<T>& entry) { return entry.first == id; });
    if (handler_to_unsubscribe_from != handler_list.end()) {
      handler_list.erase(handler_to_unsubscribe_from);
    }
  }

  void Dispatch(EventContext& event_context);
  void Flush();

  const std::vector<GameEvent>& GetEvents() const;

 private:
  template <typename T>
  using Handler = std::function<void(const T&, EventContext&)>;

  template <typename T>
  using HandlerEntry = std::pair<SubscriptionId, Handler<T>>;

  template <typename T>
  using HandlerList = std::vector<HandlerEntry<T>>;

  std::vector<GameEvent> events_;
  int next_subscription_id_ = 0;

  std::tuple<HandlerList<EnemyKilledEvent>, HandlerList<PlayerDamagedEvent>,
             HandlerList<GemCollectedEvent>,
             HandlerList<ProjectileDestroyedEvent>,
             HandlerList<ChestOpenedEvent>, HandlerList<PlayerDeadEvent>>
      handlers_;
};

}  // namespace arelto

#endif
