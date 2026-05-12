// include/observation_manager.h
#ifndef RL2_OBSERVATION_MANAGER_H_
#define RL2_OBSERVATION_MANAGER_H_

#include "constants/map.h"
#include "entity.h"
#include "map.h"
#include "scene.h"

namespace arelto {

class ObservationManager {

 public:
  int GetObservationSize();
  void UpdateObservations(Scene& scene);
  void FillObservationBuffer(float* buffer_ptr, int buffer_size,
                             const Scene& scene);

 private:
  int physics_tick_count_ = 0;

  void UpdateWorldOccupancyMap(
      FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map,
      Player& player, Enemy& enemy, Projectiles& projectiles);
  void UpdateEnemyRayCaster(
      Enemy& enemy,
      const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map);
};

}  // namespace arelto

#endif
