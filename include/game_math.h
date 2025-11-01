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

float get_length_vector2d(Vector2D vector);
float calculate_distance_vector2d(Vector2D v0, Vector2D v1);
Vector2D get_centroid(Vector2D position, Size size);

std::vector<CollisionPair> find_collisions_sap(
    Vector2D player_position, std::array<Vector2D, kNumEnemies> enemy_position);

void handle_collisions_sap(Player& player, Enemy& enemy);
std::vector<CollisionPair> get_collision_pairs_sap(
    std::array<AABB, kNumEntities> sorted_aabb);
void resolve_collision_pairs_sap(Player& player, Enemy& enemy,
                                 std::array<AABB, kNumEntities> entity_aabb,
                                 std::vector<CollisionPair> collision_pairs);

void handle_player_oob(Player& player);
void handle_enemy_oob(Enemy& enemy);

}  // namespace rl2

#endif
