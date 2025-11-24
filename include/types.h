// include/types.h
#ifndef RL2_TYPES_H_
#define RL2_TYPES_H_
#include <SDL2/SDL_render.h>
#include <constants.h>

namespace rl2 {

struct Vector2D {
  float x;
  float y;

  float Norm() const { return std::sqrt(x * x + y * y); };
  Vector2D Normalized() const {
    float n = Norm();
    if (n == 0.0f) {
      return {0.0f, 0.0f};
    }
    return {x / n, y / n};
  };
};

struct VertexData {
  SDL_Vertex vertex;
};

enum class EntityType : int {
  entity = -1,
  player = 0,
  enemy = 1,
  projectile = 2
};

// entity idx is used during collisions to handle each type of entity differently
struct AABB {
  float min_x, min_y, max_x, max_y;
  EntityType entity_type;
  int storage_index;
};

struct CollisionPair {
  int index_a, index_b;
  EntityType type_a, type_b;
};

enum class CollisionType : int {
  None = -1,
  player_terrain = 0,
  enemy_terrain = 1,
  projectile_terrain = 2,
  player_enemy = 3,
  enemy_enemy = 4,
  player_projectile = 5,
  enemy_projectile = 6,
};

}  // namespace rl2
#endif
