// include/entity.h
#ifndef RL2_ENTITY_H_
#define RL2_ENTITY_H_
#include <array>
#include <cstdint>
#include <optional>
#include <unordered_set>
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
  // TODO: Replace entity type such that only player has this type
  EntityType entity_type = EntityType::player;
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
  int proj_id;
};

class Projectiles {
 public:
  std::vector<int> owner_id_;
  std::vector<Vector2D> position_;
  std::vector<Vector2D> direction_;
  std::vector<float> speed_;
  std::vector<Size> size_;
  std::vector<float> inv_mass_;
  std::vector<int> proj_id_;
  std::unordered_set<int> to_be_destroyed_;
  EntityType entity_type_ = EntityType::projectile;
  size_t GetNumProjectiles() { return owner_id_.size(); };
  void AddProjectile(ProjectileData proj);
  void DestroyProjectile(int idx);
  void DestroyProjectiles();
};

class Player : public Entity {

 public:
  SpellStats spell_stats_;
  Fireball fireball_;
  Frostbolt frostbolt_;
  void UpdateAllSpellStats();
  std::optional<ProjectileData> CastProjectileSpell(BaseProjectileSpell& spell,
                                                    float time,
                                                    Vector2D cursor_position);
};

struct Enemies {
  std::array<bool, kNumEnemies> is_alive;
  std::array<Vector2D, kNumEnemies> position;
  std::array<Vector2D, kNumEnemies> velocity;
  std::array<int, kNumEnemies> health_points;
  std::array<float, kNumEnemies> movement_speed;
  std::array<Size, kNumEnemies> size;
  std::array<float, kNumEnemies> inv_mass;
  EntityType entity_type = EntityType::enemy;
};

void UpdateEnemyStatus(Enemies& enemies);

}  // namespace rl2

#endif
