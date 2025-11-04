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

  SDL_Texture* GetTileTexture(const char* file, SDL_Renderer* renderer);
  void SetupTileMap();
  void SetupTiles();
  void SetupTileSelector();
};

}  // namespace rl2

#endif
