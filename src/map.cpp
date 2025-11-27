// src/map.cpp
#include "map.h"
#include <SDL2/SDL.h>
#include <SDL_render.h>
#include <SDL_surface.h>
#include <math.h>
#include "constants.h"
#include "random.h"

namespace rl2 {

SDL_Texture* TileManager::GetTileTexture(const char* file,
                                         SDL_Renderer* renderer) {
  SDL_Surface* tile_map_surface = SDL_LoadBMP(file);
  SDL_Texture* tile_texture =
      SDL_CreateTextureFromSurface(renderer, tile_map_surface);
  SDL_FreeSurface(tile_map_surface);
  return tile_texture;
};

void TileManager::SetupTileMap() {
  for (int i = 0; i < kNumTilesX; ++i) {
    for (int j = 0; j < kNumTilesY; ++j) {
      tile_map_[i][j] = GenerateRandomInt(0, kNumTileTypes - 1);
    };
  };
};
void TileManager::SetupTiles() {
  for (int i = 0; i < kNumTilesX; ++i) {
    for (int j = 0; j < kNumTilesY; ++j) {
      tiles_[i][j].x = i * kTileWidth;
      tiles_[i][j].y = j * kTileHeight;
      tiles_[i][j].w = kTileWidth;
      tiles_[i][j].h = kTileHeight;
    };
  };
};
void TileManager::SetupTileSelector() {

  const int kTilesInRow = static_cast<int>(std::sqrt(kNumTileTypes));
  // const int kTilesInRow = 4;
  // const int kTilesInCol = 2;
  SDL_Rect selector;
  for (int i = 0; i < kNumTileTypes; ++i) {
    int col = i % kTilesInRow;
    int row = i / kTilesInRow;

    selector.x = col * kTileWidth;
    selector.y = row * kTileHeight;
    selector.w = kTileWidth;
    selector.h = kTileHeight;

    select_tiles_.push_back(selector);
  }
};

}  // namespace rl2
