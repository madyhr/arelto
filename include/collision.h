// include/collision.h
#ifndef RL2_COLLISION_H_
#define RL2_COLLISION_H_
#include <SDL2/SDL_render.h>
#include <constants.h>
#include <sys/types.h>
#include <array>
#include <vector>
#include "entity.h"
#include "types.h"

namespace rl2 {

std::vector<CollisionPair> FindCollisionsSAP(
    Vector2D player_position, std::array<Vector2D, kNumEnemies> enemy_position);

void HandleCollisionsSAP(Player& player, Enemy& enemy,
                         Projectiles& projectiles);
std::vector<CollisionPair> GetCollisionPairsSAP(std::vector<AABB>& sorted_aabb);
CollisionType GetCollisionType(const CollisionPair& cp);
void ResolveCollisionPairsSAP(Player& player, Enemy& enemy,
                              Projectiles& projectiles,
                              std::vector<AABB>& entity_aabb,
                              std::vector<CollisionPair>& collision_pairs);
void ResolveEnemyProjectileCollision(const CollisionPair& cp, Enemy& enemy,
                                     Projectiles& projectiles, Player& player);

void ResolveEnemyEnemyCollision(const CollisionPair& cp, Enemy& enemy,
                                std::vector<AABB>& entity_aabb);
void ResolvePlayerEnemyCollision(const CollisionPair& cp, Player& player,
                                 Enemy& enemy,
                                 std::vector<AABB>& entities_aabb);
std::array<Vector2D, 2> GetDisplacementVectors(
    const std::array<AABB, 2>& aabbs, const std::array<Vector2D, 2>& centroids,
    const std::array<float, 2>& inv_masses);

}  // namespace rl2

#endif
