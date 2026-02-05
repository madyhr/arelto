// include/physics_manager.h
#ifndef RL2_PHYSICS_MANAGER_H_
#define RL2_PHYSICS_MANAGER_H_
#include <SDL2/SDL_render.h>
#include "collision_manager.h"
#include "constants/map.h"
#include "entity.h"
#include "map.h"
#include "scene.h"

namespace arelto {

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
};

}  // namespace arelto

#endif
