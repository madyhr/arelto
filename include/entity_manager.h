#ifndef RL2_ENTITY_MANAGER_H_
#define RL2_ENTITY_MANAGER_H_

#include "constants/map.h"
#include "entity.h"
#include "event_manager.h"
#include "map.h"
#include "scene.h"

namespace arelto {

class EntityManager {
 public:
  EntityManager();
  ~EntityManager();

  void Update(Scene& scene, float dt, EventManager& event_manager);
  bool IsPlayerDead(const Player& player);

 private:
  int physics_tick_count_ = 0;

  void UpdateEnemyStatus(Scene& scene, float dt);
  void UpdateProjectilesStatus(Scene& scene, EventManager& event_manager);
  void UpdateGemStatus(Scene& scene);
  void UpdateChestStatus(Scene& scene);
  void UpdateObservations(Scene& scene, float dt);
  void UpdateWorldOccupancyMap(
      FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map,
      Player& player, Enemy& enemy, Projectiles& projectiles);
  void UpdateEnemyRayCaster(
      Enemy& enemy,
      const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map);
};

}  // namespace arelto

#endif
