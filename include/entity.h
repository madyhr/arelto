// include/entity.h
#ifndef RL2_ENTITY_H_
#define RL2_ENTITY_H_
#include <array>
#include <optional>
#include <unordered_set>
#include <vector>
#include "abilities.h"
#include "constants/enemy.h"
#include "constants/projectile.h"
#include "ray_caster.h"
#include "types.h"

namespace arelto {

// forward declaration to avoid circular dependency
class SpellManager;

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
  std::vector<int> proj_type_;
  std::unordered_set<int> to_be_destroyed_;
  EntityType entity_type_ = EntityType::projectile;
  size_t num_sprite_cells_ = kProjectileNumSpriteCells;
  size_t GetNumProjectiles() const { return owner_id_.size(); };
  void AddProjectile(ProjectileData proj);
  void DestroyProjectile(int idx);
  void DestroyProjectiles();
  void ResetAllProjectiles();
};

struct Enemy {
  std::array<bool, kNumEnemies> is_alive;
  std::array<Vector2D, kNumEnemies> position;
  std::array<Vector2D, kNumEnemies> prev_position;
  std::array<Vector2D, kNumEnemies> velocity;
  std::array<Vector2D, kNumEnemies> prev_velocity;
  std::array<int, kNumEnemies> max_health_points;
  std::array<int, kNumEnemies> health_points;
  std::array<float, kNumEnemies> movement_speed;
  std::array<Size2D, kNumEnemies> sprite_size;
  std::array<Collider, kNumEnemies> collider;
  std::array<float, kNumEnemies> inv_mass;
  std::array<float, kNumEnemies> last_horizontal_velocity;
  std::array<int, kNumEnemies> attack_damage;
  std::array<float, kNumEnemies> attack_cooldown_s;
  std::array<float, kNumEnemies> attack_cooldown_timer;
  std::array<int, kNumEnemies> damage_dealt_sim_step;
  // this is used for RL training and signifies the end of an episode if true.
  std::array<bool, kNumEnemies> is_done;
  // this is used for enemy terminations that happen due to a timeout
  std::array<bool, kNumEnemies> is_truncated_latched;
  // this is used for enemy terminations that happen before a timeout
  std::array<bool, kNumEnemies> is_terminated_latched;

  EnemyRayCaster ray_caster;
  EntityType entity_type = EntityType::enemy;
};

enum ItemId : int;

struct InventoryItem {
  ItemId item_id;
  int count;
  InventoryItem(ItemId id, int count) : item_id(id), count(count) {}
};

using Inventory = std::vector<InventoryItem>;

class Player {
 public:
  EntityType entity_type_ = EntityType::player;
  bool is_alive_ = true;
  Stats stats_;
  Vector2D position_;
  Vector2D prev_position_;
  Vector2D velocity_;
  bool is_invulnerable;
  float invulnerable_timer;
  float invulnerable_window_s_ = 0.1f;
  float exp_required_scale_ = 1.1f;
  AABB hitbox_aabb_;
  float last_horizontal_velocity_;
  SpellStats spell_stats_;
  SpellManager* spell_manager_ = nullptr;
  void SetSpellManager(SpellManager* spell_manager) {
    spell_manager_ = spell_manager;
  };
  void UpdateAllSpellStats();
  void ResetSpellsToBase();
  std::optional<ProjectileData> CastProjectileSpell(BaseProjectileSpell& spell,
                                                    float time,
                                                    Vector2D cursor_position);
  BaseProjectileSpell* GetSpell(SpellId id);
  const BaseProjectileSpell* GetSpell(SpellId id) const;
  void TakeDamage(int damage);
  void TakeHealing(int healing);
  int CalculateOutgoingDamage(float damage);
  bool IsSpellReady(BaseSpell& spell, float time);
  void AddToInventory(ItemId item_id);
  Inventory inventory_;
};

class ExpGem {
 public:
  std::vector<Rarity> rarity_;
  std::vector<Vector2D> position_;
  std::vector<Vector2D> prev_position_;
  std::vector<Collider> collider_;
  std::vector<float> inv_mass_;
  std::vector<Size2D> sprite_size_;
  std::unordered_set<int> to_be_destroyed_;
  EntityType entity_type_ = EntityType::exp_gem;

  size_t GetNumExpGems() const { return rarity_.size(); };
  void AddExpGem(ExpGemData exp_gem_data);
  void DestroyExpGem(int idx);
  void DestroyExpGems();
  void ResetAllExpGems();
};

class Chest {
 public:
  std::vector<Vector2D> position_;
  std::vector<Vector2D> prev_position_;
  std::vector<Collider> collider_;
  std::vector<float> inv_mass_;
  std::vector<Size2D> sprite_size_;
  std::unordered_set<int> to_be_destroyed_;
  EntityType entity_type_ = EntityType::chest;
  size_t GetNumChests() const { return position_.size(); };
  void AddChest(ChestData chest_data);
  void DestroyChest(int idx);
  void DestroyChests();
  void ResetAllChests();
};

Vector2D GetCentroid(const Vector2D& position, const Size2D& size);
AABB GetAABB(const Vector2D& position, const Size2D& size,
             const EntityType& type = EntityType::None,
             const int& storage_index = 0);
AABB GetCollisionAABB(const Vector2D& centroid, const Size2D& size,
                      const EntityType& type = EntityType::None,
                      const int& storage_index = 0);
void RespawnEnemyAtIndex(Enemy& enemy, const Player& player, int idx);
void SpawnAllEnemies(Enemy& enemy, const Player& player);
}  // namespace arelto

#endif
