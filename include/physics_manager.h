// include/physics_manager.h
#ifndef RL2_PHYSICS_MANAGER_H_
#define RL2_PHYSICS_MANAGER_H_
#include <SDL2/SDL_render.h>
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
  float physics_dt_;
  void UpdatePlayerState(Player& player);
  void UpdateEnemyState(Enemy& enemy, const Player& player);
  void UpdateProjectileState(Projectiles& projectiles);
  void HandleCollisions(Player& player, Enemy& enemy, Projectiles& projectiles);
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
};

}  // namespace rl2

#endif
