// src/collision_manager.cpp
#include "collision_manager.h"
#include <SDL_rect.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include "abilities.h"
#include "constants.h"
#include "entity.h"
#include "scene.h"
#include "types.h"

namespace rl2 {

void CollisionManager::HandleCollisionsSAP(Scene& scene) {
  size_t total_entities =
      1 + kNumEnemies + scene.projectiles.GetNumProjectiles();

  if (entity_aabb_.capacity() < total_entities) {
    entity_aabb_.reserve(total_entities * 2);
  };

  entity_aabb_.clear();
  scene.player.UpdateAABB();
  entity_aabb_.push_back(scene.player.aabb_);
  for (int i = 0; i < kNumEnemies; ++i) {
    entity_aabb_.push_back(GetAABB(scene.enemy.position[i], scene.enemy.size[i],
                                   scene.enemy.entity_type, i + 1));
  }
  for (int i = 0; i < scene.projectiles.GetNumProjectiles(); ++i) {
    entity_aabb_.push_back(
        GetAABB(scene.projectiles.position_[i], scene.projectiles.size_[i],
                scene.projectiles.entity_type_, i + 1 + kNumEnemies));
  }
  std::sort(entity_aabb_.begin(), entity_aabb_.end(),
            [](const AABB& a, const AABB& b) { return a.min_x < b.min_x; });

  FindCollisionPairsSAP(entity_aabb_);
  ResolveCollisionPairsSAP(scene);
};

void CollisionManager::FindCollisionPairsSAP(std::vector<AABB>& sorted_aabb) {

  collision_pairs_.clear();
  size_t count = sorted_aabb.size();
  for (int i = 0; i < count; ++i) {
    const AABB& current_aabb = sorted_aabb[i];

    for (int j = i + 1; j < count; ++j) {
      const AABB& active_aabb = sorted_aabb[j];

      // As the AABB vector is sorted by min_x, we know that there can be no collision
      // if the min of the active AABB is larger than the max of the current AABB.
      if (active_aabb.min_x > current_aabb.max_x) {
        break;
      }
      bool has_y_overlap = current_aabb.max_y > active_aabb.min_y &&
                           current_aabb.min_y < active_aabb.max_y;
      if (has_y_overlap) {
        collision_pairs_.push_back(
            {current_aabb.storage_index, active_aabb.storage_index,
             current_aabb.entity_type, active_aabb.entity_type});
      };
    }
  }
};

void CollisionManager::ResolveCollisionPairsSAP(Scene& scene) {
  for (const CollisionPair& cp : collision_pairs_) {
    CollisionType collision_type = GetCollisionType(cp);

    switch (collision_type) {
      case CollisionType::None:
        continue;
      case CollisionType::player_terrain:
        continue;
      case CollisionType::enemy_terrain:
        continue;
      case CollisionType::projectile_terrain:
        continue;
      case CollisionType::player_enemy:
        ResolvePlayerEnemyCollision(cp, scene.player, scene.enemy);
        continue;
      case CollisionType::enemy_enemy:
        ResolveEnemyEnemyCollision(cp, scene.enemy);
        continue;
      case CollisionType::player_projectile:
        continue;
      case CollisionType::enemy_projectile:
        ResolveEnemyProjectileCollision(cp, scene.enemy, scene.projectiles,
                                        scene.player);
        continue;
    }
  }
};

CollisionType CollisionManager::GetCollisionType(const CollisionPair& cp) {
  EntityType type_a = cp.type_a;
  EntityType type_b = cp.type_b;

  // Order the two types by smallest value first to simplify collison type check
  if (type_a > type_b) {
    std::swap(type_a, type_b);
  };

  if (type_a == EntityType::player && type_b == EntityType::enemy) {
    return CollisionType::player_enemy;
  } else if (type_a == EntityType::enemy && type_b == EntityType::enemy) {
    return CollisionType::enemy_enemy;
  } else if (type_a == EntityType::enemy && type_b == EntityType::projectile) {
    return CollisionType::enemy_projectile;
  } else {
    return CollisionType::None;
  }
};

std::array<Vector2D, 2> CollisionManager::GetDisplacementVectors(
    const std::array<AABB, 2>& aabbs, const std::array<Vector2D, 2>& centroids,
    const std::array<float, 2>& inv_masses) {
  std::array<Vector2D, 2> displacement_vectors;

  float overlap_x = std::min(aabbs[0].max_x, aabbs[1].max_x) -
                    std::max(aabbs[0].min_x, aabbs[1].min_x);
  float overlap_y = std::min(aabbs[0].max_y, aabbs[1].max_y) -
                    std::max(aabbs[0].min_y, aabbs[1].min_y);

  // Choose smaller axis
  bool resolve_x = (overlap_x < overlap_y);

  float push_factor = inv_masses[1] / (inv_masses[0] + inv_masses[1]);

  float direction = resolve_x
                        ? (centroids[1].x >= centroids[0].x ? 1.0f : -1.0f)
                        : (centroids[1].y >= centroids[0].y ? 1.0f : -1.0f);

  float overlap = resolve_x ? overlap_x : overlap_y;

  float ax = resolve_x ? -direction * overlap * (1.0f - push_factor) : 0.0f;
  float ay = resolve_x ? 0.0f : -direction * overlap * (1.0f - push_factor);

  float bx = resolve_x ? direction * overlap * push_factor : 0.0f;
  float by = resolve_x ? 0.0f : direction * overlap * push_factor;

  displacement_vectors[0] = {ax, ay};
  displacement_vectors[1] = {bx, by};
  return displacement_vectors;
}

void CollisionManager::ResolveEnemyEnemyCollision(const CollisionPair& cp,
                                                  Enemy& enemy) {

  int enemy_idx_a = cp.index_a - 1;
  int enemy_idx_b = cp.index_b - 1;

  float inv_mass_a = enemy.inv_mass[enemy_idx_a];
  float inv_mass_b = enemy.inv_mass[enemy_idx_b];
  Vector2D centroid_a =
      GetCentroid(enemy.position[enemy_idx_a], enemy.size[enemy_idx_a]);
  Vector2D centroid_b =
      GetCentroid(enemy.position[enemy_idx_b], enemy.size[enemy_idx_b]);

  AABB aabb_a = GetAABB(enemy.position[enemy_idx_a], enemy.size[enemy_idx_a]);
  AABB aabb_b = GetAABB(enemy.position[enemy_idx_b], enemy.size[enemy_idx_b]);

  std::array<Vector2D, 2> displacement_vectors = GetDisplacementVectors(
      {aabb_a, aabb_b},
      {centroid_a, centroid_b}, {inv_mass_a, inv_mass_b});

  enemy.position[enemy_idx_a] += displacement_vectors[0];
  enemy.position[enemy_idx_b] += displacement_vectors[1];
};

void CollisionManager::ResolvePlayerEnemyCollision(const CollisionPair& cp,
                                                   Player& player,
                                                   Enemy& enemy) {

  bool a_is_player = cp.type_a == EntityType::player;
  int player_idx = a_is_player ? cp.index_a : cp.index_b;
  int enemy_idx = a_is_player ? cp.index_b - 1 : cp.index_a - 1;

  float player_inv_mass = player.stats_.inv_mass;
  float enemy_inv_mass = enemy.inv_mass[enemy_idx];

  Vector2D player_centroid = GetCentroid(player.position_, player.stats_.size);
  Vector2D enemy_centroid =
      GetCentroid(enemy.position[enemy_idx], enemy.size[enemy_idx]);

  player.UpdateAABB();
  AABB enemy_aabb = GetAABB(enemy.position[enemy_idx], enemy.size[enemy_idx]);

  std::array<Vector2D, 2> displacement_vectors = GetDisplacementVectors(
      {player.aabb_, enemy_aabb},
      {player_centroid, enemy_centroid}, {player_inv_mass, enemy_inv_mass});

  player.position_ += displacement_vectors[0];
  enemy.position[enemy_idx] += displacement_vectors[1];

  player.stats_.health -= 1;
};

void CollisionManager::ResolveEnemyProjectileCollision(const CollisionPair& cp,
                                                       Enemy& enemy,
                                                       Projectiles& projectiles,
                                                       Player& player) {
  bool a_is_proj = cp.type_a == EntityType::projectile;
  int proj_idx =
      a_is_proj ? cp.index_a - 1 - kNumEnemies : cp.index_b - 1 - kNumEnemies;
  int enemy_idx = a_is_proj ? cp.index_b - 1 : cp.index_a - 1;
  projectiles.to_be_destroyed_.insert(proj_idx);
  int proj_id = projectiles.proj_id_[proj_idx];
  int spell_damage = player.spell_stats_.damage[proj_id];
  enemy.health_points[enemy_idx] -= spell_damage;
};

}  // namespace rl2
