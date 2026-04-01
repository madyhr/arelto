#ifndef RL2_ENTITY_MANAGER_H_
#define RL2_ENTITY_MANAGER_H_

#include <vector>
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

  void Initialize(Scene& scene, EventManager& event_manager);
  void Update(Scene& scene, float dt, EventManager& event_manager);
  void ProcessPendingSpawns(Scene& scene);

 private:
  int physics_tick_count_ = 0;
  std::vector<ExpGemData> pending_gem_spawns_;
  std::vector<ChestData> pending_chest_spawns_;

  void UpdatePlayerStatus(Scene& scene, float dt, EventManager& event_manager);
  void UpdateEnemyStatus(Scene& scene, float dt, EventManager& event_manager);
  void UpdateProjectilesStatus(Scene& scene, EventManager& event_manager);
  void UpdateGemStatus(Scene& scene);
  void UpdateChestStatus(Scene& scene);
  void UpdateObservations(Scene& scene);
  void UpdateWorldOccupancyMap(
      FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map,
      Player& player, Enemy& enemy, Projectiles& projectiles);
  void UpdateEnemyRayCaster(
      Enemy& enemy,
      const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map);
};

}  // namespace arelto

#endif
