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
  std::array<EntityType, kTotalCells> data_;
  FixedMap() { Clear(); };
  virtual ~FixedMap() {};

  void Clear() { data_.fill(EntityType::None); };

  int GetDataIdx(int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height) {
      return -1;
    };
    return x + y * width;
  };

  inline EntityType GetUnchecked(int x, int y) const {
    return data_[x + y * width];
  }

  EntityType Get(int x, int y) const {
    int data_idx = GetDataIdx(x, y);
    if (data_idx == -1) {
      return EntityType::None;
    }
    return data_[data_idx];
  };

  inline void Set(int x, int y, EntityType type) {
    int data_idx = GetDataIdx(x, y);

    if (data_idx != -1) {
      data_[data_idx] = type;
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
    for (size_t x = 0; x < width; ++x) {
      data_[x] = type;                         // Top (y=0)
      data_[x + (height - 1) * width] = type;  // Bottom
    }
    for (size_t y = 0; y < height; ++y) {
      data_[y * width] = type;              // Left (x=0)
      data_[width - 1 + y * width] = type;  // Right
    }
  }

  EntityType* Data() { return data_.data(); };
  const EntityType* Data() const { return data_.data(); };

  template <size_t src_w, size_t src_h>
  void CopyRowFrom(const FixedMap<src_w, src_h>& source, int dest_row_idx,
                   int src_row_idx, int src_col_offset) {
    EntityType* dest_ptr = Data() + dest_row_idx * width;

    if (src_row_idx < 0 || src_row_idx >= static_cast<int>(src_h)) {
      std::fill(dest_ptr, dest_ptr + width, EntityType::None);
      return;
    }

    const EntityType* src_ptr = source.Data() + src_row_idx * src_w;

    int start_copy_local = std::max(0, -src_col_offset);
    int end_copy_local = std::min(static_cast<int>(width),
                                  static_cast<int>(src_w) - src_col_offset);

    int copy_len = end_copy_local - start_copy_local;

    if (start_copy_local > 0) {
      std::fill(dest_ptr, dest_ptr + start_copy_local, EntityType::None);
    }

    if (copy_len > 0) {
      std::copy(src_ptr + src_col_offset + start_copy_local,
                src_ptr + src_col_offset + end_copy_local,
                dest_ptr + start_copy_local);
    }

    if (end_copy_local < static_cast<int>(width)) {
      std::fill(dest_ptr + end_copy_local, dest_ptr + width, EntityType::None);
    };
  };
};

}  // namespace arelto

#endif
