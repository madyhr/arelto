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

std::vector<CollisionPair> find_collisions_sap(
    Vector2D player_position, std::array<Vector2D, kNumEnemies> enemy_position);

void resolve_collisions_sap(Player& player, Enemy& enemy);

float get_length_vector2d(Vector2D vector);
float calculate_distance_vector2d(Vector2D v0, Vector2D v1);

}  // namespace rl2

#endif
