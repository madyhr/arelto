// include/entity.h
#ifndef RL2_ENTITY_H_
#define RL2_ENTITY_H_
#include <array>
#include <cstdint>
#include <vector>
#include "abilities.h"
#include "constants.h"
#include "types.h"

namespace rl2 {

struct Size {
  uint32_t width;
  uint32_t height;
};

struct Stats {
  uint32_t health;
  float movement_speed;
  Size size;
  float inv_mass;
};

class Entity {
 public:
  Stats stats_;
  Vector2D position_;
  Vector2D velocity_;
  AABB aabb_;
  void UpdateAABB();
};

struct ProjectileData {
  int owner_id;
  Vector2D position;
  Vector2D velocity;
  float speed;
  Size size;
  float inv_mass;
  int texture_id;
};

class Projectiles {
 public:
  std::vector<int> owner_ids_;
  std::vector<Vector2D> positions_;
  std::vector<Vector2D> velocities_;
  std::vector<float> speeds_;
  std::vector<Size> sizes_;
  std::vector<float> inv_masses_;
  std::vector<int> texture_ids_;
  size_t GetNumProjectiles() { return owner_ids_.size(); };
  void AddProjectile(ProjectileData proj);
  void DestroyProjectile(int idx);
};

class Player : public Entity {

 public:
  Fireball fireball_ = {0.5f, 0.0f};
  Frostbolt frostbolt_ = {0.5, 0.0f};
  ProjectileData CastSpell();
};

struct Enemies {
  std::array<bool, kNumEnemies> are_alive;
  std::array<Vector2D, kNumEnemies> positions;
  std::array<Vector2D, kNumEnemies> velocities;
  std::array<uint32_t, kNumEnemies> health_points;
  std::array<float, kNumEnemies> movement_speeds;
  std::array<Size, kNumEnemies> sizes;
  std::array<float, kNumEnemies> inv_masses;
};
}  // namespace rl2

#endif
