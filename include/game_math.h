// include/game_math.h
#ifndef RL2_GAME_MATH_H_
#define RL2_GAME_MATH_H_
#include <SDL2/SDL_render.h>
#include <constants.h>

namespace rl2 {

struct Vector2D {
  float x;
  float y;
};

struct VertexData {
  SDL_Vertex vertex;
};

}  // namespace rl2

#endif
