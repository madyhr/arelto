// src/map.cpp
#include "map.h"
#include <SDL2/SDL.h>
#include <SDL_render.h>
#include <SDL_surface.h>
#include "constants.h"
#include "random.h"

namespace rl2 {

SDL_Texture* TileManager::get_tile_texture(const char* file,
                                           SDL_Renderer* renderer) {
  SDL_Surface* tile_map_surface = SDL_LoadBMP(file);
  SDL_Texture* tile_texture =
      SDL_CreateTextureFromSurface(renderer, tile_map_surface);
  SDL_FreeSurface(tile_map_surface);
  return tile_texture;
};

void TileManager::setup_tile_map() {
  for (int i = 0; i < kNumTilesX; ++i) {
    for (int j = 0; j < kNumTilesY; ++j) {
      tile_map_[i][j] = generate_random_int(1, 4);
    };
  };
};
void TileManager::setup_tiles() {
  for (int i = 0; i < kNumTilesX; ++i) {
    for (int j = 0; j < kNumTilesY; ++j) {
      tiles_[i][j].x = i * kTileSize;
      tiles_[i][j].y = j * kTileSize;
      tiles_[i][j].w = kTileSize;
      tiles_[i][j].h = kTileSize;
    };
  };
};
void TileManager::setup_tile_selector() {

  tile_selector_.select_tile_1.x = 0;
  tile_selector_.select_tile_1.y = 0;
  tile_selector_.select_tile_1.w = kTileSize;
  tile_selector_.select_tile_1.h = kTileSize;

  tile_selector_.select_tile_2.x = kTileSize;
  tile_selector_.select_tile_2.y = 0;
  tile_selector_.select_tile_2.w = kTileSize;
  tile_selector_.select_tile_2.h = kTileSize;

  tile_selector_.select_tile_3.x = 0;
  tile_selector_.select_tile_3.y = kTileSize;
  tile_selector_.select_tile_3.w = kTileSize;
  tile_selector_.select_tile_3.h = kTileSize;

  tile_selector_.select_tile_4.x = kTileSize;
  tile_selector_.select_tile_4.y = kTileSize;
  tile_selector_.select_tile_4.w = kTileSize;
  tile_selector_.select_tile_4.h = kTileSize;
};

}  // namespace rl2
