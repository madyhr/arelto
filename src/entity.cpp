// src/entity.cpp
#include "entity.h"
#include <algorithm>
#include "constants/enemy.h"
#include "constants/map.h"
#include "random.h"
#include "spell_manager.h"
#include "types.h"

namespace arelto {

Vector2D GetCentroid(const Vector2D& position, const Size2D& size) {
  return {position.x + 0.5f * static_cast<float>(size.width),
          position.y + 0.5f * static_cast<float>(size.height)};
}

AABB GetAABB(const Vector2D& position, const Size2D& size,
             const EntityType& type, const int& storage_index) {
  return {position.x,
          position.y,
          position.x + static_cast<float>(size.width),
          position.y + static_cast<float>(size.height),
          type,
          storage_index};
};

AABB GetCollisionAABB(const Vector2D& centroid, const Size2D& size,
                      const EntityType& type, const int& storage_index) {

  float half_w = 0.5f * static_cast<float>(size.width);
  float half_h = 0.5f * static_cast<float>(size.height);

  return {centroid.x - half_w,
          centroid.y - half_h,
          centroid.x + half_w,
          centroid.y + half_h,
          type,
          storage_index};
};

void RespawnEnemyAtIndex(Enemy& enemy, const Player& player, int idx) {
  int max_x = kMapWidth - static_cast<int>(enemy.sprite_size[idx].width);
  int max_y = kMapHeight - static_cast<int>(enemy.sprite_size[idx].height);

  Vector2D potential_pos;
  do {
    potential_pos = {(float)GenerateRandomInt(0, max_x),
                     (float)GenerateRandomInt(0, max_y)};
  } while ((potential_pos - player.position_).Norm() <
           kEnemyMinimumInitialDistance);

  enemy.position[idx] = potential_pos;
  enemy.prev_position[idx] = potential_pos;
  enemy.prev_velocity[idx] = {0.0f, 0.0f};
  enemy.health_points[idx] = enemy.max_health_points[idx];
  enemy.damage_dealt_sim_step[idx] = 0;
  enemy.is_alive[idx] = true;
  enemy.is_done[idx] = false;
}

void SpawnAllEnemies(Enemy& enemy, const Player& player) {
  for (int i = 0; i < kNumEnemies; ++i) {
    bool enemy_needs_spawn = !enemy.is_alive[i] || enemy.is_done[i];
    if (enemy_needs_spawn) {
      RespawnEnemyAtIndex(enemy, player, i);
    }
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
    owner_id_[idx] = owner_id_[last_idx];
    position_[idx] = position_[last_idx];
    prev_position_[idx] = prev_position_[last_idx];
    direction_[idx] = direction_[last_idx];
    speed_[idx] = speed_[last_idx];
    collider_[idx] = collider_[last_idx];
    sprite_size_[idx] = sprite_size_[last_idx];
    inv_mass_[idx] = inv_mass_[last_idx];
    proj_type_[idx] = proj_type_[last_idx];
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
  if (!spell_manager_) {
    return;
  }
  for (const auto& spell : spell_manager_->GetAllSpells()) {
    spell->SetTimeOfLastUse(-1.0f);
    spell_stats_.SetProjectileSpellStats(*spell);
  }
};

void Player::ResetSpellsToBase() {
  if (!spell_manager_) {
    return;
  }
  spell_manager_->ResetSpellStats();
}

std::optional<ProjectileData> Player::CastProjectileSpell(
    BaseProjectileSpell& spell, float time, Vector2D cursor_position) {

  if (!IsSpellReady(spell, time)) {
    return std::nullopt;
  }
  Vector2D player_centroid = GetCentroid(position_, stats_.size.GetSize());
  Vector2D spell_direction = (cursor_position - player_centroid).Normalized();
  Size2D spell_size = spell.GetSpriteSize();
  Vector2D spell_position = player_centroid - (ToVector2D(spell_size) / 2.0f);
  ProjectileData projectile_spell = {static_cast<int>(entity_type_),
                                     spell_position,
                                     spell_direction,
                                     spell.GetSpeed(),
                                     spell.GetCollider(),
                                     spell_size,
                                     spell.GetInvMass(),
                                     spell.GetId()};
  spell.SetTimeOfLastUse(time);
  return projectile_spell;
};

BaseProjectileSpell* Player::GetSpell(SpellId id) {
  if (!spell_manager_) {
    return nullptr;
  }
  return spell_manager_->GetSpell(id);
}

const BaseProjectileSpell* Player::GetSpell(SpellId id) const {
  if (!spell_manager_) {
    return nullptr;
  }
  return spell_manager_->GetSpell(id);
}

void Player::TakeDamage(int damage) {
  int raw_damage = damage;
  int armor = static_cast<int>(stats_.armor.GetValue());
  int final_damage = std::max(0, raw_damage - armor);
  stats_.health -= final_damage;
};

void Player::TakeHealing(int healing) {
  int raw_healing = healing;
  int max_health = static_cast<int>(stats_.max_health.GetValue());
  stats_.health = std::min(max_health, stats_.health + raw_healing);
};

void Player::AddToInventory(ItemId item_id) {
  for (auto& item : inventory_) {
    if (item.item_id == item_id) {
      item.count++;
      return;
    }
  }
  inventory_.emplace_back(item_id, 1);
}

int Player::CalculateOutgoingDamage(float damage) {
  float global_dmg_mod = stats_.global_damage_modifier.GetValue();
  int total_damage = static_cast<int>(std::round(damage * global_dmg_mod));
  return total_damage;
}

bool Player::IsSpellReady(BaseSpell& spell, float time) {
  float modified_cd =
      spell.GetCooldown() * stats_.global_cooldown_modifier.GetValue();

  return time >= (spell.GetTimeOfLastUse() + modified_cd);
}

void ExpGem::AddExpGem(ExpGemData gem) {
  rarity_.push_back(gem.rarity);
  position_.push_back(gem.position);
  // upon initialization prev pos should be set to initial pos to
  // avoid errors during render interpolation.
  prev_position_.push_back(gem.position);
  collider_.push_back(gem.collider);
  inv_mass_.push_back(gem.inv_mass);
  sprite_size_.push_back(gem.sprite_size);
};

void ExpGem::DestroyExpGem(int idx) {
  size_t last_idx = position_.size() - 1;
  if (idx != last_idx) {
    rarity_[idx] = rarity_[last_idx];
    position_[idx] = position_[last_idx];
    prev_position_[idx] = prev_position_[last_idx];
    collider_[idx] = collider_[last_idx];
    inv_mass_[idx] = inv_mass_[last_idx];
    sprite_size_[idx] = sprite_size_[last_idx];
  }

  rarity_.pop_back();
  position_.pop_back();
  prev_position_.pop_back();
  collider_.pop_back();
  inv_mass_.pop_back();
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
  inv_mass_.clear();
  to_be_destroyed_.clear();
};

void Chest::AddChest(ChestData chest) {
  position_.push_back(chest.position);
  prev_position_.push_back(chest.position);
  collider_.push_back(chest.collider);
  inv_mass_.push_back(chest.inv_mass);
  sprite_size_.push_back(chest.sprite_size);
};

void Chest::DestroyChest(int idx) {
  size_t last_idx = position_.size() - 1;
  if (idx != last_idx) {
    position_[idx] = position_[last_idx];
    prev_position_[idx] = prev_position_[last_idx];
    collider_[idx] = collider_[last_idx];
    inv_mass_[idx] = inv_mass_[last_idx];
    sprite_size_[idx] = sprite_size_[last_idx];
  }
  position_.pop_back();
  prev_position_.pop_back();
  collider_.pop_back();
  inv_mass_.pop_back();
  sprite_size_.pop_back();
};

void Chest::DestroyChests() {
  if (to_be_destroyed_.empty()) {
    return;
  };
  std::vector<int> sorted_indices(to_be_destroyed_.begin(),
                                  to_be_destroyed_.end());
  std::sort(sorted_indices.begin(), sorted_indices.end(), std::greater<int>());
  for (int idx : sorted_indices) {
    DestroyChest(idx);
  };
  to_be_destroyed_.clear();
};

void Chest::ResetAllChests() {
  position_.clear();
  prev_position_.clear();
  collider_.clear();
  inv_mass_.clear();
  sprite_size_.clear();
  to_be_destroyed_.clear();
};

}  // namespace arelto
