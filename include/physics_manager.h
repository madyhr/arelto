// include/physics_manager.h
#ifndef RL2_PHYSICS_MANAGER_H_
#define RL2_PHYSICS_MANAGER_H_
#include <SDL2/SDL_render.h>
#include "collision_manager.h"
#include "constants.h"
#include "entity.h"
#include "map.h"
#include "scene.h"

namespace rl2 {

class PhysicsManager {

 public:
  PhysicsManager();
  ~PhysicsManager();

  bool Initialize();
  void StepPhysics(Scene& scene);
  float GetPhysicsDt() { return physics_dt_; };
  void SetPhysicsDt(float dt) { physics_dt_ = dt; };

 private:
  int tick_count_ = 0;
  float physics_dt_;
  CollisionManager collision_manager_;
  void UpdatePlayerState(Player& player);
  void UpdateEnemyState(Enemy& enemy, const Player& player);
  void UpdateProjectileState(Projectiles& projectiles);
  void HandleCollisions(Scene& scene);
  void HandleOutOfBounds(Player& player, Enemy& enemy,
                         Projectiles& projectiles);
  void HandlePlayerOOB(Player& player);
  void HandleEnemyOOB(Enemy& enemy);
  void HandleProjectileOOB(Projectiles& projectiles);
  void UpdateWorldOccupancyMap(
      FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map,
      Player& player, Enemy& enemy, Projectiles& projectiles);
  void UpdateEnemyOccupancyMap(
      Enemy& enemy,
      FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map);
  void UpdateEnemyStatus(Enemy& enemy, const Player& player);
  void UpdateProjectilesStatus(Projectiles& projectiles);
};

}  // namespace rl2

#endif
