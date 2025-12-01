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

class Camera {
 public:
  Vector2D position_;
  Vector2D prev_position_;
  Vector2D render_position_;
};

inline auto WorldToGrid(auto pos) {
  return (pos / kOccupancyMapResolution);
}
}  // namespace rl2
#endif
