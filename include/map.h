// include/map.h
#ifndef RL2_MAP_H_
#define RL2_MAP_H_
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_surface.h>
#include <vector>
#include "constants/map.h"
#include "types.h"

namespace arelto {

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

template <size_t width, size_t height>
class FixedMap {
  // Example with width = 3, height = 2;
  // data idx of 0, 1, 2, 3, 4, 5
  // maps to -->
  // 0 1 2
  // 3 4 5

 public:
  static constexpr size_t kTotalCells = width * height;
  // Data contains `EntityType`s stored as a bitmask.
  std::array<uint16_t, kTotalCells> data_;
  FixedMap() { Clear(); };
  virtual ~FixedMap() {};

  void Clear() { data_.fill(kMaskTypeNone); };

  int GetDataIdx(int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height) {
      return -1;
    };
    return x + y * width;
  };

  inline EntityType GetUnchecked(int x, int y) const {
    return MaskToEntityTypePrioritized(data_[x + y * width]);
  }

  EntityType Get(int x, int y) const {
    int data_idx = GetDataIdx(x, y);
    if (data_idx == -1) {
      return EntityType::None;
    }
    return MaskToEntityTypePrioritized(data_[data_idx]);
  };

  uint16_t GetMask(int x, int y) const {
    int data_idx = GetDataIdx(x, y);
    if (data_idx == -1) {
      return kMaskTypeNone;
    }
    return data_[data_idx];
  }

  // This functions directly sets the entire type bitmask for a
  // grid cell to just that of the specified EntityType.
  inline void Set(int x, int y, EntityType type) {
    int data_idx = GetDataIdx(x, y);

    if (data_idx != -1) {
      data_[data_idx] = EntityTypeToMask(type);
    }
  }

  inline void Add(int x, int y, EntityType type) {
    int data_idx = GetDataIdx(x, y);
    if (data_idx != -1) {
      data_[data_idx] |= EntityTypeToMask(type);
    }
  }

  inline void Remove(int x, int y, EntityType type) {
    int data_idx = GetDataIdx(x, y);
    if (data_idx != -1) {
      data_[data_idx] &= ~EntityTypeToMask(type);
    }
  }

  inline void ClearCell(int x, int y) {
    int data_idx = GetDataIdx(x, y);
    if (data_idx != -1) {
      data_[data_idx] = kMaskTypeNone;
    }
  }

  inline void SetGrid(int x, int y, int w, int h, EntityType type) {
    // Sets a grid of cells to a certain entity type.
    for (int i = 0; i < w + 1; ++i) {
      for (int j = 0; j < h + 1; ++j) {
        Set(x + i, y + j, type);
      }
    }
  }

  void AddBorder(EntityType type) {
    uint16_t mask = EntityTypeToMask(type);
    for (size_t x = 0; x < width; ++x) {
      data_[x] |= mask;                         // Top (y=0)
      data_[x + (height - 1) * width] |= mask;  // Bottom
    }
    for (size_t y = 0; y < height; ++y) {
      data_[y * width] |= mask;              // Left (x=0)
      data_[width - 1 + y * width] |= mask;  // Right
    }
  }

  uint16_t* Data() { return data_.data(); };
  const uint16_t* Data() const { return data_.data(); };
};

}  // namespace arelto

#endif
