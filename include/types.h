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
};

struct VertexData {
  SDL_Vertex vertex;
};

struct AABB {
  float min_x, min_y, max_x, max_y;
  int entity_idx;
};

struct CollisionPair {
  int index_a, index_b;
};

}  // namespace rl2
#endif
