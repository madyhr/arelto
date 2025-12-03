// include/entity.h
#ifndef RL2_ENTITY_H_
#define RL2_ENTITY_H_
#include <array>
#include <optional>
#include <unordered_set>
#include <vector>
#include "abilities.h"
#include "constants.h"
#include "map.h"
#include "types.h"

namespace rl2 {

class Projectiles {
 public:
  std::vector<int> owner_id_;
  std::vector<Vector2D> position_;
  std::vector<Vector2D> prev_position_;
  std::vector<Vector2D> direction_;
  std::vector<float> speed_;
  std::vector<Size2D> sprite_size_;
  std::vector<Collider> collider_;
  std::vector<float> inv_mass_;
  std::vector<int> proj_id_;
  std::unordered_set<int> to_be_destroyed_;
  EntityType entity_type_ = EntityType::projectile;
  size_t GetNumProjectiles() const { return owner_id_.size(); };
  void AddProjectile(ProjectileData proj);
  void DestroyProjectile(int idx);
  void DestroyProjectiles();
};

struct Enemy {
  std::array<bool, kNumEnemies> is_alive;
  std::array<Vector2D, kNumEnemies> position;
  std::array<Vector2D, kNumEnemies> prev_position;
  std::array<Vector2D, kNumEnemies> velocity;
  std::array<int, kNumEnemies> health_points;
  std::array<float, kNumEnemies> movement_speed;
  std::array<Size2D, kNumEnemies> sprite_size;
  std::array<Collider, kNumEnemies> collider;
  std::array<float, kNumEnemies> inv_mass;
  std::array<float, kNumEnemies> last_horizontal_velocity;
  std::array<FixedMap<kEnemyOccupancyMapWidth, kEnemyOccupancyMapHeight>,
             kNumEnemies>
      occupancy_map;
  EntityType entity_type = EntityType::enemy;
};

class Player {
 public:
  EntityType entity_type_ = EntityType::player;
  Stats stats_;
  Vector2D position_;
  Vector2D prev_position_;
  Vector2D velocity_;
  Collider collider_;
  bool is_invulnerable;
  float invulnerable_timer;
  AABB hitbox_aabb_;
  float last_horizontal_velocity_;
  SpellStats<kNumPlayerSpells> spell_stats_;
  Fireball fireball_;
  Frostbolt frostbolt_;
  void UpdateAllSpellStats();
  std::optional<ProjectileData> CastProjectileSpell(BaseProjectileSpell& spell,
                                                    float time,
                                                    Vector2D cursor_position);
};

Vector2D GetCentroid(Vector2D position, Size2D size);
AABB GetAABB(Vector2D position, Size2D size, EntityType type = EntityType::None,
             int storage_index = 0);
AABB GetCollisionAABB(Vector2D centroid, Size2D size,
                      EntityType type = EntityType::None, int storage_index = 0);
void UpdateEnemyStatus(Enemy& enemies, const Player& player);
void UpdateProjectilesStatus(Projectiles& projectiles);
void RespawnEnemy(Enemy& enemy, const Player& player);

}  // namespace rl2

#endif
