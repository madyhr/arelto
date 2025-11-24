// include/game_math.h
#ifndef RL2_GAME_MATH_H_
#define RL2_GAME_MATH_H_
#include <SDL2/SDL_render.h>
#include <constants.h>
#include <sys/types.h>
#include <array>
#include <vector>
#include "entity.h"
#include "types.h"

namespace rl2 {

Vector2D SubtractVector2D(Vector2D v0, Vector2D v1);

float CalculateVector2DDistance(Vector2D v0, Vector2D v1);
Vector2D GetCentroid(Vector2D position, Size size);

void HandlePlayerOOB(Player& player);
void HandleEnemyOOB(Enemy& enemy);
void HandleProjectileOOB(Projectiles& projectiles);
void DestroyProjectiles(Projectiles& projectiles);

}  // namespace rl2

#endif
