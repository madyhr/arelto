// include/types.h
#ifndef RL2_TYPES_H_
#define RL2_TYPES_H_
#include <SDL2/SDL_render.h>
#include <array>
#include <vector>
#include "constants.h"

namespace rl2 {

struct Vector2D {
  float x;
  float y;

  float Norm() const { return std::sqrt(x * x + y * y); };
  Vector2D Normalized() const {
    // Calculating the inverse norm to use 1 division instead of 2.
    float n = Norm();
    if (n < 1e-6f) {
      return {0.0f, 0.0f};
    }
    float inv_n = 1.0f / n;
    return {x * inv_n, y * inv_n};
  };

  Vector2D& operator+=(const Vector2D& other) {
    x += other.x;
    y += other.y;
    return *this;
  }
  Vector2D& operator-=(const Vector2D& other) {
    x -= other.x;
    y -= other.y;
    return *this;
  }
};

// Vector2D & Vector2D
inline bool operator==(const Vector2D& vector0, const Vector2D& vector1) {
  return vector0.x == vector1.x && vector0.y == vector1.y;
};

inline Vector2D operator+(const Vector2D& vector0, const Vector2D& vector1) {
  return {vector0.x + vector1.x, vector0.y + vector1.y};
};

inline Vector2D operator-(const Vector2D& vector0, const Vector2D& vector1) {
  return {vector0.x - vector1.x, vector0.y - vector1.y};
};

// Vector2D & float
inline Vector2D operator+(const Vector2D& vector, const float& scalar) {
  return {vector.x + scalar, vector.y + scalar};
};

inline Vector2D operator+(const float& scalar, const Vector2D& vector) {
  return {vector.x + scalar, vector.y + scalar};
};

inline Vector2D operator-(const Vector2D& vector, const float& scalar) {
  return {vector.x - scalar, vector.y - scalar};
};

inline Vector2D operator-(const float& scalar, const Vector2D& vector) {
  return {vector.x - scalar, vector.y - scalar};
};

inline Vector2D operator*(const float& scalar, const Vector2D& vector) {
  return {vector.x * scalar, vector.y * scalar};
};

inline Vector2D operator*(const Vector2D& vector, const float& scalar) {
  return {vector.x * scalar, vector.y * scalar};
};

inline Vector2D operator/(const Vector2D& vector, float scalar) {
  // Calculating the inverse scalar to avoid using division twice.
  float inv_scalar = 1.0f / scalar;
  return {vector.x * inv_scalar, vector.y * inv_scalar};
}



inline Vector2D LerpVector2D(const Vector2D& start, const Vector2D& end,
                             const float& alpha) {
  return start * (1 - alpha) + end * alpha;
};

struct VertexData {
  SDL_Vertex vertex;
};

enum class EntityType : int { None = -1, terrain, player, enemy, projectile };

// entity type is used during collisions to handle each type of entity differently
// storage index is used to be able to retrieve the original index of the entity
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
  None = 0,
  player_terrain,
  enemy_terrain,
  projectile_terrain,
  player_enemy,
  enemy_enemy,
  player_projectile,
  enemy_projectile,
};

struct EntityPosition {
  Vector2D position;
  EntityType type;
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
  virtual ~FixedMap(){};

  void Clear() { data_.fill(EntityType::None); };

  int GetDataIdx(int x, int y) {
    if (x < 0 || x > width || y < 0 || y > height) {
      return -1;
    };
    return x + y * width;
  };

  inline EntityType Get(int x, int y) {
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

}  // namespace rl2
#endif
