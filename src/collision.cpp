// src/collision.cpp
#include "collision.h"
#include <SDL_rect.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include "abilities.h"
#include "constants.h"
#include "entity.h"
#include "game_math.h"
#include "types.h"

namespace rl2 {

void HandleCollisionsSAP(Player& player, Enemy& enemies,
                         Projectiles& projectiles) {
  std::vector<AABB> entities_aabb;
  player.UpdateAABB();
  entities_aabb.push_back(player.aabb_);
  for (int i = 0; i < kNumEnemies; ++i) {
    entities_aabb.push_back({enemies.position[i].x, enemies.position[i].y,
                             enemies.position[i].x + enemies.size[i].width,
                             enemies.position[i].y + enemies.size[i].height,
                             enemies.entity_type, i + 1});
  }
  for (int i = 0; i < projectiles.GetNumProjectiles(); ++i) {
    entities_aabb.push_back(
        {projectiles.position_[i].x, projectiles.position_[i].y,
         projectiles.position_[i].x + projectiles.size_[i].width,
         projectiles.position_[i].y + projectiles.size_[i].height,
         projectiles.entity_type_, i + 1 + kNumEnemies});
  }
  std::vector<AABB> sorted_aabb = entities_aabb;
  std::sort(sorted_aabb.begin(), sorted_aabb.end(),
            [](const AABB& a, const AABB& b) { return a.min_x < b.min_x; });

  std::vector<CollisionPair> collision_pairs =
      GetCollisionPairsSAP(sorted_aabb);
  ResolveCollisionPairsSAP(player, enemies, projectiles, entities_aabb,
                           collision_pairs);
};

std::vector<CollisionPair> GetCollisionPairsSAP(
    std::vector<AABB>& sorted_aabb) {
  std::vector<CollisionPair> collision_pairs;
  // std::sort(sorted_aabb.begin(), sorted_aabb.end(),
  //           [](const AABB& a, const AABB& b) { return a.min_x < b.min_x; });
  std::vector<const AABB*> active_list;
  for (int i = 0; i < sorted_aabb.size(); ++i) {
    const AABB& current_aabb = sorted_aabb[i];

    // Prune
    active_list.erase(std::remove_if(active_list.begin(), active_list.end(),
                                     [&](const AABB* active_aabb) {
                                       return active_aabb->max_x <
                                              current_aabb.min_x;
                                     }),
                      active_list.end());
    // Search
    for (const AABB* active_aabb : active_list) {
      bool has_y_overlap = current_aabb.max_y > active_aabb->min_y &&
                           current_aabb.min_y < active_aabb->max_y;
      if (has_y_overlap) {
        collision_pairs.push_back(
            {current_aabb.storage_index, active_aabb->storage_index,
             current_aabb.entity_type, active_aabb->entity_type});
      };
    }

    // Add
    active_list.push_back(&current_aabb);
  }

  return collision_pairs;
};

void ResolveCollisionPairsSAP(Player& player, Enemy& enemy,
                              Projectiles& projectiles,
                              std::vector<AABB>& entities_aabb,
                              std::vector<CollisionPair>& collision_pairs) {
  for (const CollisionPair& cp : collision_pairs) {
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
        ResolvePlayerEnemyCollision(cp, player, enemy, entities_aabb);
        continue;
      case CollisionType::enemy_enemy:
        ResolveEnemyEnemyCollision(cp, enemy, entities_aabb);
        continue;
      case CollisionType::player_projectile:
        continue;
      case CollisionType::enemy_projectile:
        ResolveEnemyProjectileCollision(cp, enemy, projectiles, player);
        continue;
    }

    const AABB& a = entities_aabb[cp.index_a];
    const AABB& b = entities_aabb[cp.index_b];

    float overlap_x = std::min(a.max_x, b.max_x) - std::max(a.min_x, b.min_x);
    float overlap_y = std::min(a.max_y, b.max_y) - std::max(a.min_y, b.min_y);

    if (overlap_x <= 0.0f || overlap_y <= 0.0f)
      continue;  // No overlap

    // Choose smaller axis
    bool resolve_x = (overlap_x < overlap_y);

    auto MoveEntity = [&](EntityType entity_type, int idx, float dx, float dy) {
      if (entity_type == EntityType::player) {
        player.position_.x += dx;
        player.position_.y += dy;
      } else if (entity_type == EntityType::enemy) {
        int enemy_idx = idx - 1;
        enemy.position[enemy_idx].x += dx;
        enemy.position[enemy_idx].y += dy;
      }
    };
    auto GetEntityCentroid = [&](EntityType entity_type, int idx) -> Vector2D {
      if (entity_type == EntityType::player) {
        return rl2::GetCentroid(player.position_, player.stats_.size);
      } else {
        int enemy_idx = idx - 1;
        return rl2::GetCentroid(enemy.position[enemy_idx],
                                enemy.size[enemy_idx]);
      }
    };
    auto GetEntityInvMass = [&](EntityType entity_type, int idx) -> float {
      if (entity_type == EntityType::player) {
        return player.stats_.inv_mass;
      } else {
        int enemy_idx = idx - 1;
        return enemy.inv_mass[enemy_idx];
      }
    };
    float inv_mass_a = GetEntityInvMass(cp.type_a, cp.index_a);
    float inv_mass_b = GetEntityInvMass(cp.type_b, cp.index_b);
    float push_factor = inv_mass_b / (inv_mass_a + inv_mass_b);

    Vector2D centroid_a = GetEntityCentroid(cp.type_a, cp.index_a);
    Vector2D centroid_b = GetEntityCentroid(cp.type_b, cp.index_b);

    float direction = resolve_x ? (centroid_b.x >= centroid_a.x ? 1.0f : -1.0f)
                                : (centroid_b.y >= centroid_a.y ? 1.0f : -1.0f);

    float overlap = resolve_x ? overlap_x : overlap_y;

    float ax = resolve_x ? -direction * overlap * (1.0f - push_factor) : 0.0f;
    float ay = resolve_x ? 0.0f : -direction * overlap * (1.0f - push_factor);

    float bx = resolve_x ? direction * overlap * (1.0f - push_factor) : 0.0f;
    float by = resolve_x ? 0.0f : direction * overlap * (1.0f - push_factor);

    MoveEntity(cp.type_a, cp.index_a, ax, ay);
    MoveEntity(cp.type_b, cp.index_b, bx, by);
  }
};

CollisionType GetCollisionType(const CollisionPair& cp) {
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

std::array<Vector2D, 2> GetDisplacementVectors(
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

void ResolveEnemyEnemyCollision(const CollisionPair& cp, Enemy& enemy,
                                std::vector<AABB>& entities_aabb) {

  int enemy_idx_a = cp.index_a - 1;
  int enemy_idx_b = cp.index_b - 1;

  float inv_mass_a = enemy.inv_mass[enemy_idx_a];
  float inv_mass_b = enemy.inv_mass[enemy_idx_b];
  Vector2D centroid_a =
      GetCentroid(enemy.position[enemy_idx_a], enemy.size[enemy_idx_a]);
  Vector2D centroid_b =
      GetCentroid(enemy.position[enemy_idx_b], enemy.size[enemy_idx_b]);

  std::array<Vector2D, 2> displacement_vectors = GetDisplacementVectors(
      {entities_aabb[cp.index_a], entities_aabb[cp.index_b]},
      {centroid_a, centroid_b}, {inv_mass_a, inv_mass_b});

  enemy.position[enemy_idx_a].x += displacement_vectors[0].x;
  enemy.position[enemy_idx_a].y += displacement_vectors[0].y;
  enemy.position[enemy_idx_b].x += displacement_vectors[1].x;
  enemy.position[enemy_idx_b].y += displacement_vectors[1].y;
};

void ResolvePlayerEnemyCollision(const CollisionPair& cp, Player& player,
                                 Enemy& enemy,
                                 std::vector<AABB>& entities_aabb) {

  bool a_is_player = cp.type_a == EntityType::player;
  int player_idx = a_is_player ? cp.index_a : cp.index_b;
  int enemy_idx = a_is_player ? cp.index_b - 1 : cp.index_a - 1;

  float inv_mass_player = player.stats_.inv_mass;
  float inv_mass_enemy = enemy.inv_mass[enemy_idx];

  Vector2D centroid_player = GetCentroid(player.position_, player.stats_.size);
  Vector2D centroid_enemy =
      GetCentroid(enemy.position[enemy_idx], enemy.size[enemy_idx]);

  std::array<Vector2D, 2> displacement_vectors = GetDisplacementVectors(
      {entities_aabb[player_idx], entities_aabb[enemy_idx + 1]},
      {centroid_player, centroid_enemy}, {inv_mass_player, inv_mass_enemy});

  player.position_.x += displacement_vectors[0].x;
  player.position_.y += displacement_vectors[0].y;
  enemy.position[enemy_idx].x += displacement_vectors[1].x;
  enemy.position[enemy_idx].y += displacement_vectors[1].y;
};


void ResolveEnemyProjectileCollision(const CollisionPair& cp, Enemy& enemy,
                                     Projectiles& projectiles, Player& player) {
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
