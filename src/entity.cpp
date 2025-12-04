// src/entity.cpp
#include "entity.h"
#include <algorithm>
#include "constants.h"
#include "random.h"
#include "types.h"

namespace rl2 {

Vector2D GetCentroid(Vector2D position, Size2D size) {
  return {position.x + 0.5f * size.width, position.y + 0.5f * size.height};
}

AABB GetAABB(Vector2D position, Size2D size, EntityType type,
             int storage_index) {
  return {position.x,
          position.y,
          position.x + size.width,
          position.y + size.height,
          type,
          storage_index};
};

AABB GetCollisionAABB(Vector2D centroid, Size2D size, EntityType type,
                      int storage_index) {

  float half_w = 0.5 * size.width;
  float half_h = 0.5 * size.height;

  return {centroid.x - half_w,
          centroid.y - half_h,
          centroid.x + half_w,
          centroid.y + half_h,
          type,
          storage_index};
};

void UpdateEnemyStatus(Enemy& enemies, const Player& player) {
  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemies.health_points[i] <= 0) {
      enemies.is_alive[i] = false;
      RespawnEnemy(enemies, player);
    };
  };
};

void UpdateProjectilesStatus(Projectiles& projectiles) {
  projectiles.DestroyProjectiles();
};

void RespawnEnemy(Enemy& enemy, const Player& player) {

  int max_x = kMapWidth - kEnemySpriteWidth;
  int max_y = kMapHeight - kEnemySpriteHeight;

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
    enemy.prev_position[i] = potential_pos;
    enemy.health_points[i] = kEnemyHealth;
    enemy.is_alive[i] = true;
  };
};

void Projectiles::AddProjectile(ProjectileData proj) {
  owner_id_.push_back(proj.owner_id);
  position_.push_back(proj.position);
  // upon initialization prev pos should be set to initial pos to
  // avoid errors during render interpolation.
  prev_position_.push_back(proj.position);
  direction_.push_back(proj.velocity);
  speed_.push_back(proj.speed);
  collider_.push_back(proj.collider);
  sprite_size_.push_back(proj.sprite_size);
  inv_mass_.push_back(proj.inv_mass);
  proj_id_.push_back(proj.proj_id);
};

void Projectiles::DestroyProjectile(int idx) {
  size_t last_idx = position_.size() - 1;
  if (idx != last_idx) {
    owner_id_[idx] = std::move(owner_id_[last_idx]);
    position_[idx] = std::move(position_[last_idx]);
    prev_position_[idx] = std::move(prev_position_[last_idx]);
    direction_[idx] = std::move(direction_[last_idx]);
    speed_[idx] = std::move(speed_[last_idx]);
    collider_[idx] = std::move(collider_[last_idx]);
    sprite_size_[idx] = std::move(sprite_size_[last_idx]);
    inv_mass_[idx] = std::move(inv_mass_[last_idx]);
    proj_id_[idx] = std::move(proj_id_[last_idx]);
  }

  owner_id_.pop_back();
  position_.pop_back();
  prev_position_.pop_back();
  direction_.pop_back();
  speed_.pop_back();
  collider_.pop_back();
  sprite_size_.pop_back();
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

void Projectiles::ResetAllProjectiles() {
  owner_id_.clear();
  position_.clear();
  prev_position_.clear();
  direction_.clear();
  speed_.clear();
  sprite_size_.clear();
  collider_.clear();
  inv_mass_.clear();
  proj_id_.clear();
  to_be_destroyed_.clear();
};

void Player::UpdateAllSpellStats() {
  fireball_.SetTimeOfLastUse(-1.0f);
  frostbolt_.SetTimeOfLastUse(-1.0f);
  spell_stats_.SetProjectileSpellStats(fireball_);
  spell_stats_.SetProjectileSpellStats(frostbolt_);
};

std::optional<ProjectileData> Player::CastProjectileSpell(
    BaseProjectileSpell& spell, float time, Vector2D cursor_position) {

  bool spell_is_ready = time >= spell.GetReadyTime();
  if (spell_is_ready) {
    Vector2D centroid = GetCentroid(position_, stats_.sprite_size);
    Vector2D spell_direction = (cursor_position - centroid).Normalized();
    ProjectileData projectile_spell = {static_cast<int>(entity_type_),
                                       position_,
                                       spell_direction,
                                       spell.GetSpeed(),
                                       spell.GetCollider(),
                                       spell.GetSpriteSize(),
                                       spell.GetInvMass(),
                                       spell.GetId()};
    spell.SetTimeOfLastUse(time);
    return projectile_spell;
  }
  return std::nullopt;
};

}  // namespace rl2
