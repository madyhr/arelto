// include/types.h
#ifndef RL2_TYPES_H_
#define RL2_TYPES_H_
#include <cstdint>
#include <numbers>
#include "constants/map.h"
#include "utils.h"

namespace arelto {

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

// Bitmask constants for EntityType
constexpr uint16_t kMaskTypeNone = 0;
constexpr uint16_t kMaskTypeTerrain = 1 << 0;
constexpr uint16_t kMaskTypePlayer = 1 << 1;
constexpr uint16_t kMaskTypeEnemy = 1 << 2;
constexpr uint16_t kMaskTypeProjectile = 1 << 3;
constexpr uint16_t kMaskTypeExpGem = 1 << 4;

inline uint16_t EntityTypeToMask(EntityType type) {
  switch (type) {
    case EntityType::terrain:
      return kMaskTypeTerrain;
    case EntityType::player:
      return kMaskTypePlayer;
    case EntityType::enemy:
      return kMaskTypeEnemy;
    case EntityType::projectile:
      return kMaskTypeProjectile;
    case EntityType::exp_gem:
      return kMaskTypeExpGem;
    default:
      return kMaskTypeNone;
  }
}

// This function returns the *highest priority* EntityType in the
// mask and only that EntityType.
// Note: Order is paramount here, so use this with care.
inline EntityType MaskToEntityTypePrioritized(uint16_t mask) {
  if (mask & kMaskTypePlayer)
    return EntityType::player;
  if (mask & kMaskTypeTerrain)
    return EntityType::terrain;
  if (mask & kMaskTypeEnemy)
    return EntityType::enemy;
  if (mask & kMaskTypeProjectile)
    return EntityType::projectile;
  if (mask & kMaskTypeExpGem)
    return EntityType::exp_gem;
  return EntityType::None;
}

// A bitmask of the EntityType's that block ray casts.
// This is used in the case a single ray can register multiple hits and
// continue until it reaches a blocking EntityType.
constexpr uint16_t kMaskRayHitBlockingTypes =
    kMaskTypePlayer | kMaskTypeTerrain;

// A bitmask of the EntityType's that do not block ray casts.
// Note: Currently only the first non-blocking hit is registered.
constexpr uint16_t kMaskRayHitNonBlockingTypes = kMaskTypeProjectile;

// This function returns the highest priority blocking entity in the mask.
inline EntityType MaskToBlockingType(uint16_t mask) {
  return MaskToEntityTypePrioritized(mask & kMaskRayHitBlockingTypes);
}

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
  bool is_headless;
  bool show_occupancy_map = false;
  bool show_ray_caster = false;
};

enum GameState : int {
  in_start_screen = 0,
  in_main_menu,
  is_running,
  is_gameover,
  in_shutdown,
  in_settings_menu,
  in_level_up,
};

enum Rarity : int { common, rare, epic, legendary, Count };

struct ExpGemData {
  Rarity rarity;
  Vector2D position;
  Vector2D prev_position;
  Collider collider;
  Size2D sprite_size;
};

enum class UpgradeType : int { damage = 0, speed, cooldown, size, count };

}  // namespace arelto
#endif
