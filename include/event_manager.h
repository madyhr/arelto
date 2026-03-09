// include/event_manager.h
#ifndef RL2_EVENT_MANAGER_H_
#define RL2_EVENT_MANAGER_H_

#include <cstdint>
#include <vector>

namespace arelto {

enum class EventType : uint8_t {
  enemy_killed,
  player_damaged,
  gem_collected,
  projectile_destroyed,
};

struct GameEvent {
  EventType type;
  int entity_index;     // primary entity (enemy, gem, projectile index)
  int secondary_index;  // related entity (e.g. projectile index for a kill)
  float value;          // contextual value (damage amount, XP value, etc.)
};

class EventManager {
 public:
  void Emit(GameEvent event);
  void Flush();

  const std::vector<GameEvent>& GetEvents() const;
  std::vector<GameEvent> GetEvents(EventType type) const;

 private:
  std::vector<GameEvent> events_;
};

}  // namespace arelto

#endif
