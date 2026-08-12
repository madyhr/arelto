// include/types.h
#ifndef RL2_TYPES_H_
#define RL2_TYPES_H_
#include <pybind11/attr.h>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>
#include "constants/map.h"
#include "constants/render.h"
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

inline Vector2D Lerp(const Vector2D& start, const Vector2D& end,
                     const float& alpha) {
  return start * (1 - alpha) + end * alpha;
};

inline Vector2D Round(Vector2D v) {
  return {std::nearbyint(v.x), std::nearbyint(v.y)};
}

enum class EntityType : int {
  None = 0,
  terrain,
  player,
  enemy,
  projectile,
  exp_gem,
  chest
};

// Bitmask constants for EntityType
constexpr uint16_t kMaskTypeNone = 0;
constexpr uint16_t kMaskTypeTerrain = 1 << 0;
constexpr uint16_t kMaskTypePlayer = 1 << 1;
constexpr uint16_t kMaskTypeEnemy = 1 << 2;
constexpr uint16_t kMaskTypeProjectile = 1 << 3;
constexpr uint16_t kMaskTypeExpGem = 1 << 4;
constexpr uint16_t kMaskTypeChest = 1 << 5;

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
    case EntityType::chest:
      return kMaskTypeChest;
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
  if (mask & kMaskTypeChest)
    return EntityType::chest;
  return EntityType::None;
}

// A bitmask of the EntityType's that block ray casts.
// This is used in the case a single ray can register multiple hits and
// continue until it reaches a blocking EntityType.
constexpr uint16_t kMaskRayHitBlockingTypes =
    kMaskTypePlayer | kMaskTypeTerrain;

// A bitmask of the EntityType's that do not block ray casts.
// Note: Currently only the first non-blocking hit is registered.
constexpr uint16_t kMaskRayHitNonBlockingTypes =
    kMaskTypeProjectile | kMaskTypeEnemy;

// This function returns the highest priority blocking entity in the mask.
inline EntityType MaskToBlockingType(uint16_t mask) {
  return MaskToEntityTypePrioritized(mask & kMaskRayHitBlockingTypes);
}

struct Size2D {
  uint32_t width;
  uint32_t height;
};

inline Vector2D ToVector2D(const Size2D& size) {
  return {static_cast<float>(size.width), static_cast<float>(size.height)};
}

// The offset is defined as the offset from a reference position to the center
// of the Collider.
struct Collider {
  Vector2D offset;
  Size2D size;
};

// Shrink the input dimension by the input margin for more satisfying
// collision detection.
inline uint32_t ShrinkColliderDimension(uint32_t dimension, uint32_t margin) {
  if (dimension == 0) {
    return 0;
  }
  // We return 1 in this case to avoid uint32_t underflow and to ensure that
  // the collider dimension is non-zero unless explicitly set to 0.
  if (margin >= dimension) {
    return 1;
  }
  return dimension - margin;
}

inline Collider CreateCenteredCollider(
    Size2D size, uint32_t margin = kSpriteColliderMargin) {
  return Collider{{0.5f * static_cast<float>(size.width),
                   0.5f * static_cast<float>(size.height)},
                  {ShrinkColliderDimension(size.width, margin),
                   ShrinkColliderDimension(size.height, margin)}};
}

enum class ModifierType { flat = 0, percent_add, percent_mult };

struct ValueRange {
  float current;
  float updated;
};

struct Modifier {
  float value;
  ModifierType type;
  void* source;
};

class Stat {

 public:
  float GetValue() const {

    if (!is_dirty_) {
      return cached_value_;
    }

    cached_value_ = ComputeValue();
    is_dirty_ = false;

    return cached_value_;
  };

  // Returns what GetValue() would return if an additional modifier were applied,
  // without actually modifying the stat.
  float GetModifiedValue(float modifier_value,
                         ModifierType modifier_type) const {
    Modifier extra_modifier{modifier_value, modifier_type, nullptr};
    return ComputeValue(&extra_modifier);
  }

  int GetValueCeil() const { return static_cast<int>(std::ceil(GetValue())); }
  int GetValueFloor() const { return static_cast<int>(std::floor(GetValue())); }

  void SetBaseValue(float value) {
    base_value_ = value;
    is_dirty_ = true;
  }

  float GetBaseValue() { return base_value_; }

  void AddModifier(const Modifier& mod) {
    modifiers_.push_back(mod);
    is_dirty_ = true;
  }

  void ClearModifiers() {
    modifiers_.clear();
    is_dirty_ = true;
  }

 private:
  // Computes the value of the stat based on the base value and all modifiers,
  // The `extra_modifier` optional argument is used for simulating the effect
  // of a potential modifier without actually adding it to the stat.
  float ComputeValue(const Modifier* extra_modifier = nullptr) const {
    float flat_sum = 0;
    float percent_add_sum = 0;
    float percent_mult_prod = 1.0f;

    for (const Modifier& mod : modifiers_) {
      switch (mod.type) {
        case ModifierType::flat:
          flat_sum += mod.value;
          break;
        case ModifierType::percent_add:
          percent_add_sum += mod.value;
          break;
        case ModifierType::percent_mult:
          percent_mult_prod *= (1.0f + mod.value);
          break;
      }
    };

    if (extra_modifier != nullptr) {
      switch (extra_modifier->type) {
        case ModifierType::flat:
          flat_sum += extra_modifier->value;
          break;
        case ModifierType::percent_add:
          percent_add_sum += extra_modifier->value;
          break;
        case ModifierType::percent_mult:
          percent_mult_prod *= (1.0f + extra_modifier->value);
          break;
      }
    }

    return (base_value_ + flat_sum) * (1.0f + percent_add_sum) *
           percent_mult_prod;
  }

  mutable float base_value_ = 0.0f;
  std::vector<Modifier> modifiers_;
  mutable float cached_value_ = 0.0f;
  mutable bool is_dirty_ = true;
};

// A class that defines the size of an entity only by its `width`.
// Accepts an `aspect_ratio` as an input argument if width and height differ.
// Stores width as a `Stat` for easier modification of the size.
class StatsSize {

 public:
  StatsSize() = default;
  StatsSize(float aspect_ratio) : aspect_ratio_(aspect_ratio) {};

  void SetBaseWidth(float width) { width_.SetBaseValue(width); }
  uint32_t GetWidth() { return static_cast<uint32_t>(width_.GetValue()); };
  uint32_t GetWidth() const {
    return static_cast<uint32_t>(width_.GetValue());
  };
  uint32_t GetHeight() {
    return static_cast<uint32_t>(width_.GetValue() * aspect_ratio_);
  };
  uint32_t GetHeight() const {
    return static_cast<uint32_t>(width_.GetValue() * aspect_ratio_);
  };
  Size2D GetSize() { return Size2D{GetWidth(), GetHeight()}; }
  Size2D GetSize() const { return Size2D{GetWidth(), GetHeight()}; }
  Collider GetCollider() {
    return CreateCenteredCollider(GetSize(), kSpriteColliderMargin);
  };

  Stat width_;

 private:
  float aspect_ratio_ = 1.0f;
};

struct Stats {
  int level;
  int exp_points;
  Stat exp_points_required;
  int health;
  Stat max_health;
  Stat movement_speed;
  StatsSize size;
  Stat inv_mass;
  Stat armor;
  Stat global_damage_modifier;
  Stat global_cooldown_modifier;
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
  player_expgem,
  player_chest,
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
  void UpdatePosition(const Vector2D& player_centroid) {
    position_.x = player_centroid.x - 0.5f * kWindowWidth;
    position_.y = player_centroid.y - 0.5f * kWindowHeight;

    if (position_.x < 0) {
      position_.x = 0.0f;
    };
    if (position_.y < 0) {
      position_.y = 0.0f;
    };
    if (position_.x > (kMapWidth - kWindowWidth)) {
      position_.x = kMapWidth - kWindowWidth;
    }
    if (position_.y > (kMapHeight - kWindowHeight)) {
      position_.y = kMapHeight - kWindowHeight;
    }
  };
};

inline auto WorldToGrid(auto pos) {
  return (pos / kOccupancyMapResolution);
}

inline auto GridToWorld(auto pos) {
  return (pos * kOccupancyMapResolution);
}

inline float Deg2Rad(float deg) {
  return deg * static_cast<float>((std::numbers::pi / 180));
}

inline float Rad2Deg(float rad) {
  return rad * static_cast<float>((180 / std::numbers::pi));
}

struct GameStatus {
  FrameStats frame_stats;
  bool is_headless;
  bool show_occupancy_map = false;
  bool show_ray_caster = false;
};

enum class GameState : int {
  in_start_screen = 0,
  in_main_menu,
  is_running,
  is_gameover,
  in_shutdown,
  in_settings_menu,
  in_level_up,
  in_quit_confirm,
  in_chest_opening,
  in_item_selection,
};

enum class Rarity : int { common, rare, epic, legendary, count };
constexpr std::size_t to_index(Rarity rarity) {
  return static_cast<std::size_t>(rarity);
}

struct ExpGemData {
  Rarity rarity;
  Vector2D position;
  Vector2D prev_position;
  Collider collider;
  float inv_mass;
  Size2D sprite_size;
};

struct ChestData {
  Vector2D position;
  Vector2D prev_position;
  Collider collider;
  float inv_mass;
  Size2D sprite_size;
};

enum class SpellUpgradeType : int { damage = 0, speed, cooldown, size, count };

}  // namespace arelto
#endif
