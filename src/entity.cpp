// src/entity.cpp
#include "entity.h"
#include <algorithm>
#include "constants/enemy.h"
#include "constants/map.h"
#include "random.h"
#include "types.h"

namespace arelto {

Vector2D GetCentroid(const Vector2D& position, const Size2D& size) {
  return {position.x + 0.5f * size.width, position.y + 0.5f * size.height};
}

AABB GetAABB(const Vector2D& position, const Size2D& size,
             const EntityType& type, const int& storage_index) {
  return {position.x,
          position.y,
          position.x + size.width,
          position.y + size.height,
          type,
          storage_index};
};

AABB GetCollisionAABB(const Vector2D& centroid, const Size2D& size,
                      const EntityType& type, const int& storage_index) {

  float half_w = 0.5 * size.width;
  float half_h = 0.5 * size.height;

  return {centroid.x - half_w,
          centroid.y - half_h,
          centroid.x + half_w,
          centroid.y + half_h,
          type,
          storage_index};
};

void RespawnEnemy(Enemy& enemy, const Player& player) {

  int max_x = kMapWidth - kEnemySpriteWidth;
  int max_y = kMapHeight - kEnemySpriteHeight;

  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemy.is_alive[i] && !enemy.is_done[i]) {
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
    enemy.prev_velocity[i] = {0.0f, 0.0f};
    enemy.health_points[i] = kEnemyHealth;
    enemy.damage_dealt_sim_step[i] = 0;
    enemy.is_alive[i] = true;
    enemy.is_done[i] = false;
    enemy.timeout_timer[i] = 0.0f;
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
  proj_type_.push_back(proj.proj_type);
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
    proj_type_[idx] = std::move(proj_type_[last_idx]);
  }

  owner_id_.pop_back();
  position_.pop_back();
  prev_position_.pop_back();
  direction_.pop_back();
  speed_.pop_back();
  collider_.pop_back();
  sprite_size_.pop_back();
  inv_mass_.pop_back();
  proj_type_.pop_back();
};

void Projectiles::DestroyProjectiles() {
  if (to_be_destroyed_.empty()) {
    return;
  };
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
  proj_type_.clear();
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

BaseProjectileSpell* Player::GetSpell(SpellId id) {
  switch (id) {
    case SpellId::FireballId:
      return &fireball_;
    case SpellId::FrostboltId:
      return &frostbolt_;
    default:
      return nullptr;
  }
}

const BaseProjectileSpell* Player::GetSpell(SpellId id) const {
  switch (id) {
    case SpellId::FireballId:
      return &fireball_;
    case SpellId::FrostboltId:
      return &frostbolt_;
    default:
      return nullptr;
  }
}

void ExpGem::AddExpGem(ExpGemData gem) {
  rarity_.push_back(gem.rarity);
  position_.push_back(gem.position);
  // upon initialization prev pos should be set to initial pos to
  // avoid errors during render interpolation.
  prev_position_.push_back(gem.position);
  collider_.push_back(gem.collider);
  sprite_size_.push_back(gem.sprite_size);
};

void ExpGem::DestroyExpGem(int idx) {
  size_t last_idx = position_.size() - 1;
  if (idx != last_idx) {
    rarity_[idx] = std::move(rarity_[last_idx]);
    position_[idx] = std::move(position_[last_idx]);
    prev_position_[idx] = std::move(prev_position_[last_idx]);
    collider_[idx] = std::move(collider_[last_idx]);
    sprite_size_[idx] = std::move(sprite_size_[last_idx]);
  }

  rarity_.pop_back();
  position_.pop_back();
  prev_position_.pop_back();
  collider_.pop_back();
  sprite_size_.pop_back();
};

void ExpGem::DestroyExpGems() {
  if (to_be_destroyed_.empty()) {
    return;
  };
  std::vector<int> sorted_indices(to_be_destroyed_.begin(),
                                  to_be_destroyed_.end());
  std::sort(sorted_indices.begin(), sorted_indices.end(), std::greater<int>());

  for (int idx : sorted_indices) {
    DestroyExpGem(idx);
  };

  to_be_destroyed_.clear();
};

void ExpGem::ResetAllExpGems() {
  rarity_.clear();
  position_.clear();
  prev_position_.clear();
  sprite_size_.clear();
  collider_.clear();
  to_be_destroyed_.clear();
};

}  // namespace arelto
