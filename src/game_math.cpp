// src/game_math.cpp
#include "game_math.h"
#include <SDL_rect.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "abilities.h"
#include "constants.h"
#include "entity.h"
#include "types.h"

namespace rl2 {

Vector2D SubtractVector2D(Vector2D v0, Vector2D v1) {
  return {v0.x - v1.x, v0.y - v1.y};
}

Vector2D NormalizeVector2D(Vector2D vector) {
  float norm = vector.Norm();
  vector.x = vector.x / norm;
  vector.y = vector.y / norm;
  return vector;
};

float CalculateVector2dDistance(Vector2D v0, Vector2D v1) {
  float dx = v1.x - v0.x;
  float dy = v1.y - v0.y;
  float distance = std::hypot(dx, dy);
  return distance;
};

Vector2D GetCentroid(Vector2D position, Size size) {
  return {position.x + 0.5f * size.width, position.y + 0.5f * size.height};
}

void HandleCollisionsSAP(Player& player, Enemies& enemies,
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

std::vector<CollisionPair> GetCollisionPairsSAP(std::vector<AABB> sorted_aabb) {
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

void ResolveCollisionPairsSAP(Player& player, Enemies& enemies,
                              Projectiles& projectiles,
                              std::vector<AABB> entities_aabb,
                              std::vector<CollisionPair> collision_pairs) {
  for (const CollisionPair& cp : collision_pairs) {
    CollisionType collision_type = GetCollisionType(cp);

    if (collision_type == CollisionType::None) {
      continue;
    } else if (collision_type == CollisionType::enemy_projectile) {
      bool a_is_proj = cp.type_a == EntityType::projectile;
      int proj_idx = a_is_proj ? cp.index_a - 1 - kNumEnemies
                               : cp.index_b - 1 - kNumEnemies;
      int enemy_idx = a_is_proj ? cp.index_b - 1 : cp.index_a - 1;
      projectiles.to_be_destroyed_.insert(proj_idx);
      int proj_id = projectiles.proj_id_[proj_idx];
      int spell_damage = player.spell_stats_.damage[proj_id];
      enemies.health_points[enemy_idx] -= spell_damage;
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
        enemies.position[enemy_idx].x += dx;
        enemies.position[enemy_idx].y += dy;
      }
    };

    auto GetEntityCentroid = [&](EntityType entity_type, int idx) -> Vector2D {
      if (entity_type == EntityType::player) {
        return rl2::GetCentroid(player.position_, player.stats_.size);
      } else {
        int enemy_idx = idx - 1;
        return rl2::GetCentroid(enemies.position[enemy_idx],
                                enemies.size[enemy_idx]);
      }
    };
    auto GetEntityInvMass = [&](EntityType entity_type, int idx) -> float {
      if (entity_type == EntityType::player) {
        return player.stats_.inv_mass;
      } else {
        int enemy_idx = idx - 1;
        return enemies.inv_mass[enemy_idx];
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

CollisionType GetCollisionType(CollisionPair cp) {
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

void HandlePlayerOOB(Player& player) {
  if (player.position_.x < 0) {
    player.position_.x = 0;
  }
  if (player.position_.y < 0) {
    player.position_.y = 0;
  }
  if ((player.position_.x + player.stats_.size.width) > kMapWidth) {
    player.position_.x = kMapWidth - player.stats_.size.width;
  }
  if ((player.position_.y + player.stats_.size.height) > kMapHeight) {
    player.position_.y = kMapHeight - player.stats_.size.height;
  }
};

void HandleEnemyOOB(Enemies& enemies) {
  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemies.is_alive[i]) {
      if (enemies.position[i].x < 0) {
        enemies.position[i].x = 0;
      }
      if (enemies.position[i].y < 0) {
        enemies.position[i].y = 0;
      }
      if ((enemies.position[i].x + enemies.size[i].width) > kMapWidth) {
        enemies.position[i].x = kMapWidth - enemies.size[i].width;
      }
      if ((enemies.position[i].y + enemies.size[i].height) > kMapHeight) {
        enemies.position[i].y = kMapHeight - enemies.size[i].height;
      }
    }
  }
};

void HandleProjectileOOB(Projectiles& projectiles) {
  size_t num_projectiles = projectiles.GetNumProjectiles();
  if (num_projectiles == 0) {
    return;
  }
  for (int i = 0; i < num_projectiles; ++i) {
    if (projectiles.position_[i].x < 0 || projectiles.position_[i].y < 0 ||
        (projectiles.position_[i].x + projectiles.size_[i].width) > kMapWidth ||
        (projectiles.position_[i].y + projectiles.size_[i].height) >
            kMapHeight) {
      projectiles.to_be_destroyed_.insert(i);
    }
  }
};

}  // namespace rl2
