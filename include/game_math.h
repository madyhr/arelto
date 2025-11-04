// include/game_math.h
#ifndef RL2_GAME_MATH_H_
#define RL2_GAME_MATH_H_
#include <SDL2/SDL_render.h>
#include <constants.h>
#include <sys/types.h>
#include <array>
#include <vector>
#include "entity.h"

namespace rl2 {

float CalculateVector2dDistance(Vector2D v0, Vector2D v1);
Vector2D GetCentroid(Vector2D position, Size size);

std::vector<CollisionPair> FindCollisionsSAP(
  Vector2D player_position, std::array<Vector2D, kNumEnemies> enemy_position);

void HandleCollisionsSAP(Player& player, Enemy& enemy);
std::vector<CollisionPair> GetCollisionPairsSAP(
  std::array<AABB, kNumEntities> sorted_aabb);
void ResolveCollisionPairsSAP(Player& player, Enemy& enemy,
                              std::array<AABB, kNumEntities> entity_aabb,
                              std::vector<CollisionPair> collision_pairs);

void HandlePlayerOOB(Player& player);
void HandleEnemyOOB(Enemy& enemy);

}  // namespace rl2

#endif
