// include/entity.h
#ifndef RL2_ENTITY_H_
#define RL2_ENTITY_H_
#include <array>
#include <cstdint>
#include <vector>
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
  Stats stats;
  Vector2D position;
  Vector2D velocity;
  AABB aabb;
  void update_aabb();
};

class Player : public Entity {};

struct Enemy {
  std::array<bool, kNumEnemies> is_alive;
  std::array<Vector2D, kNumEnemies> position;
  std::array<Vector2D, kNumEnemies> velocity;
  std::array<uint32_t, kNumEnemies> health;
  std::array<float, kNumEnemies> movement_speed;
  std::array<Size, kNumEnemies> size;
  std::array<float, kNumEnemies> inv_mass;
};

struct ProjectileData {
  int owner_id;
  Vector2D position;
  Vector2D velocity;
  float speed;
  Size size;
  float inv_mass;
};

class Projectiles {
  public:
    std::vector<int> owner_id;
    std::vector<Vector2D> position;
    std::vector<Vector2D> velocity;
    std::vector<float> speed;
    std::vector<Size> size;
    std::vector<float> inv_mass;
    void add_projectile(ProjectileData proj);
    void destroy_projectile(int idx);
};

}  // namespace rl2

#endif
