// src/entity.cpp
#include "entity.h"
#include <algorithm>
#include "constants.h"
#include "game_math.h"
#include "random.h"
#include "types.h"

namespace rl2 {

void UpdateEnemyStatus(Enemy& enemies, const Player& player) {
  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemies.health_points[i] <= 0) {
      enemies.is_alive[i] = false;
      RespawnEnemy(enemies, player);
    };
  };
};

void RespawnEnemy(Enemy& enemy, const Player& player) {

  int max_x = kMapWidth - kEnemyWidth;
  int max_y = kMapHeight - kEnemyHeight;

  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemy.is_alive[i]) {
      continue;
    }

    Vector2D potential_pos;

    do {
      potential_pos = {(float)GenerateRandomInt(0, max_x),
                       (float)GenerateRandomInt(0, max_y)};

    } while ((potential_pos - player.position_).Norm() <
             kEnemyMinimumInitialDistance);

    enemy.position[i] = potential_pos;
    enemy.health_points[i] = kEnemyHealth;
    enemy.is_alive[i] = true;
  };
};

void Projectiles::AddProjectile(ProjectileData proj) {
  owner_id_.push_back(proj.owner_id);
  position_.push_back(proj.position);
  direction_.push_back(proj.velocity);
  speed_.push_back(proj.speed);
  size_.push_back(proj.size);
  inv_mass_.push_back(proj.inv_mass);
  proj_id_.push_back(proj.proj_id);
};

void Projectiles::DestroyProjectile(int idx) {
  size_t last_idx = position_.size() - 1;
  if (idx != last_idx) {
    owner_id_[idx] = std::move(owner_id_[last_idx]);
    position_[idx] = std::move(position_[last_idx]);
    direction_[idx] = std::move(direction_[last_idx]);
    speed_[idx] = std::move(speed_[last_idx]);
    size_[idx] = std::move(size_[last_idx]);
    inv_mass_[idx] = std::move(inv_mass_[last_idx]);
    proj_id_[idx] = std::move(proj_id_[last_idx]);
  };
  owner_id_.pop_back();
  position_.pop_back();
  direction_.pop_back();
  speed_.pop_back();
  size_.pop_back();
  inv_mass_.pop_back();
  proj_id_.pop_back();
};

void Projectiles::DestroyProjectiles() {
  std::vector<int> sorted_indices(to_be_destroyed_.begin(),
                                  to_be_destroyed_.end());
  std::sort(sorted_indices.begin(), sorted_indices.end(), std::greater<int>());

  for (int idx : sorted_indices) {
    DestroyProjectile(idx);
  };

  to_be_destroyed_.clear();
};

void Player::UpdateAABB() {
  aabb_ = {position_.x,
           position_.y,
           position_.x + stats_.size.width,
           position_.y + stats_.size.height,
           entity_type_,
           0};
};

void Player::UpdateAllSpellStats() {
  spell_stats_.SetProjectileSpellStats(fireball_);
  spell_stats_.SetProjectileSpellStats(frostbolt_);
};

std::optional<ProjectileData> Player::CastProjectileSpell(
    BaseProjectileSpell& spell, float time, Vector2D cursor_position) {

  bool spell_is_ready = time >= spell.GetReadyTime();
  if (spell_is_ready) {
    Vector2D centroid = GetCentroid(position_, stats_.size);
    Vector2D spell_direction = (cursor_position - centroid).Normalized();
    ProjectileData projectile_spell = {(int)entity_type_,
                                       position_,
                                       spell_direction,
                                       spell.GetSpeed(),
                                       {spell.GetWidth(), spell.GetHeight()},
                                       spell.GetInvMass(),
                                       spell.GetId()};
    spell.SetTimeOfLastUse(time);
    return projectile_spell;
  }
  return std::nullopt;
};

}  // namespace rl2
