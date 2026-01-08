// include/collision_manager.h
#ifndef RL2_COLLISION_MANAGER_H_
#define RL2_COLLISION_MANAGER_H_
#include <SDL2/SDL_render.h>
#include <sys/types.h>
#include <array>
#include <vector>
#include "entity.h"
#include "scene.h"
#include "types.h"

namespace rl2 {

class CollisionManager {

 public:
  std::vector<AABB> entity_aabb_;
  std::vector<CollisionPair> collision_pairs_;
  void HandleCollisionsSAP(Scene& scene);

 private:
  void FindCollisionPairsSAP(std::vector<AABB>& sorted_aabb);
  CollisionType GetCollisionType(const CollisionPair& cp);
  void ResolveCollisionPairsSAP(Scene& scene);
  void ResolveEnemyProjectileCollision(const CollisionPair& cp, Enemy& enemy,
                                       Projectiles& projectiles,
                                       Player& player);

  void ResolveEnemyEnemyCollision(const CollisionPair& cp, Enemy& enemy);
  void ResolvePlayerEnemyCollision(const CollisionPair& cp, Player& player,
                                   Enemy& enemy);
  void ResolvePlayerGemCollision(const CollisionPair& cp,
                                 const Projectiles& projectiles,
                                 Player& player, ExpGem& exp_gem);
  std::array<Vector2D, 2> GetDisplacementVectors(
      const std::array<AABB, 2>& aabbs,
      const std::array<Vector2D, 2>& centroids,
      const std::array<float, 2>& inv_masses);
};
}  // namespace rl2

#endif
