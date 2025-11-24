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
  Vector2D Normalized() const {
    float n = Norm();
    if (n == 0.0f) {
      return {0.0f, 0.0f};
    }
    return {x / n, y / n};
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

inline Vector2D operator/(const Vector2D& vector, const float& scalar) {
  return {vector.x / scalar, vector.y / scalar};
};



struct VertexData {
  SDL_Vertex vertex;
};

enum class EntityType : int {
  entity = -1,
  player = 0,
  enemy = 1,
  projectile = 2
};

// entity idx is used during collisions to handle each type of entity differently
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
  None = -1,
  player_terrain = 0,
  enemy_terrain = 1,
  projectile_terrain = 2,
  player_enemy = 3,
  enemy_enemy = 4,
  player_projectile = 5,
  enemy_projectile = 6,
};

}  // namespace rl2
#endif
