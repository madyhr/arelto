// include/event_manager.h
#ifndef RL2_EVENT_MANAGER_H_
#define RL2_EVENT_MANAGER_H_

#include <functional>
#include <tuple>
#include <variant>
#include <vector>

namespace arelto {

class Player;  // forward declaration for EventContext

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

using GameEvent =
    std::variant<EnemyKilledEvent, PlayerDamagedEvent, GemCollectedEvent,
                 ProjectileDestroyedEvent, ChestOpenedEvent>;

// -----------------------------------------------------------------------------
// EventContext that is passed to every handler during EventManager.Dispatch
// -----------------------------------------------------------------------------

struct EventContext {
  Player& player;
};

// -----------------------------------------------------------------------------
// EventManager
// -----------------------------------------------------------------------------

class EventManager {
 public:
  void Emit(GameEvent event);

  template <typename T>
  void Subscribe(std::function<void(const T&, EventContext&)> handler) {
    std::get<std::vector<std::function<void(const T&, EventContext&)>>>(
        handlers_)
        .push_back(std::move(handler));
  }

  void Dispatch(EventContext& event_context);
  void Flush();

  const std::vector<GameEvent>& GetEvents() const;

 private:
  std::vector<GameEvent> events_;

  std::tuple<
      std::vector<std::function<void(const EnemyKilledEvent&, EventContext&)>>,
      std::vector<
          std::function<void(const PlayerDamagedEvent&, EventContext&)>>,
      std::vector<std::function<void(const GemCollectedEvent&, EventContext&)>>,
      std::vector<
          std::function<void(const ProjectileDestroyedEvent&, EventContext&)>>,
      std::vector<std::function<void(const ChestOpenedEvent&, EventContext&)>>>
      handlers_;
};

}  // namespace arelto

#endif
