// include/map.h
#ifndef RL2_MAP_H_
#define RL2_MAP_H_
#include <SDL_render.h>
#include <SDL_surface.h>
#include <vector>
#include "constants.h"

namespace rl2 {

class TileManager {
 public:
  SDL_Texture* tile_texture_;
  SDL_Surface* tile_surface_;
  SDL_Rect tiles_[kNumTilesX][kNumTilesY];
  int tile_map_[kNumTilesX][kNumTilesY];
  std::vector<SDL_Rect> select_tiles_;

  SDL_Texture* get_tile_texture(const char* file, SDL_Renderer* renderer);
  void setup_tile_map();
  void setup_tiles();
  void setup_tile_selector();
};

}  // namespace rl2

#endif
