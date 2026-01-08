// include/types.h
#ifndef RL2_TYPES_H_
#define RL2_TYPES_H_
#include <cstdint>
#include <numbers>
#include "constants/map.h"
#include "utils.h"

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
  Vector2D& operator*=(float scalar) {
    x *= scalar;
    y *= scalar;
    return *this;
  }
  Vector2D& operator/=(float scalar) {
    // Optional: Check for zero division here
    float inv_scalar = 1.0f / scalar;
    x *= inv_scalar;
    y *= inv_scalar;
    return *this;
  }
  float Dot(const Vector2D& other) const {
    return (x * other.x) + (y * other.y);
  }
  // Returns > 0 if 'other' is clockwise from 'this'
  // Returns < 0 if 'other' is counter-clockwise
  float Cross(const Vector2D& other) const {
    return (x * other.y) - (y * other.x);
  }
};

// Vector + Vector
inline Vector2D operator+(Vector2D lhs, const Vector2D& rhs) {
  lhs += rhs;
  return lhs;
}

// Vector - Vector
inline Vector2D operator-(Vector2D lhs, const Vector2D& rhs) {
  lhs -= rhs;
  return lhs;
}

// Vector * Float
inline Vector2D operator*(Vector2D lhs, float scalar) {
  lhs *= scalar;
  return lhs;
}

// Float * Vector (Commutativity)
inline Vector2D operator*(float scalar, Vector2D rhs) {
  rhs *= scalar;
  return rhs;
}

// Vector / Float
inline Vector2D operator/(Vector2D lhs, float scalar) {
  lhs /= scalar;
  return lhs;
}

inline bool operator==(const Vector2D& lhs, const Vector2D& rhs) {
  // Check if difference is negligible
  constexpr float epsilon = 1e-5f;
  return std::abs(lhs.x - rhs.x) < epsilon && std::abs(lhs.y - rhs.y) < epsilon;
}

inline bool operator!=(const Vector2D& lhs, const Vector2D& rhs) {
  return !(lhs == rhs);
}

inline Vector2D LerpVector2D(const Vector2D& start, const Vector2D& end,
                             const float& alpha) {
  return start * (1 - alpha) + end * alpha;
};

enum class EntityType : int {
  None = 0,
  terrain,
  player,
  enemy,
  projectile,
  exp_gem
};

struct Size2D {
  uint32_t width;
  uint32_t height;
};

// The offset is defined as the offset from a reference position to the center
// of the Collider.
struct Collider {
  Vector2D offset;
  Size2D size;
};

struct Stats {
  int level;
  int exp_points;
  int exp_points_required;
  int health;
  int max_health;
  float movement_speed;
  Size2D sprite_size;
  float inv_mass;
};

struct ProjectileData {
  int owner_id;
  Vector2D position;
  Vector2D velocity;
  float speed;
  Collider collider;
  Size2D sprite_size;
  float inv_mass;
  int proj_type;
};

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
  player_gem,
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

inline auto GridToWorld(auto pos) {
  return (pos * kOccupancyMapResolution);
}

inline float Deg2Rad(float deg) {
  return deg * (std::numbers::pi / 180);
}

inline float Rad2Deg(float rad) {
  return rad * (180 / std::numbers::pi);
}

struct GameStatus {
  FrameStats frame_stats;
  bool is_debug;
  bool is_headless;
};

enum GameState : int {
  in_start_screen = 0,
  in_main_menu,
  is_running,
  is_gameover,
  in_shutdown,
  is_paused,
};

enum ExpGemType : int { small, medium, large, huge };

struct ExpGemData {
  ExpGemType gem_type;
  Vector2D position;
  Vector2D prev_position;
  Collider collider;
  Size2D sprite_size;
};

}  // namespace rl2
#endif
