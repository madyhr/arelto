// src/game_math.cpp
#include "game_math.h"
#include <SDL_rect.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include "constants.h"
#include "entity.h"
#include "types.h"

namespace rl2 {

Vector2D SubtractVector2D(Vector2D v0, Vector2D v1){
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

void HandleCollisionsSAP(Player& player, Enemies& enemies) {
  std::array<AABB, kNumEntities> entities_aabb;
  player.UpdateAABB();
  entities_aabb[0] = player.aabb_;
  for (int i = 1; i < kNumEnemies + 1; ++i) {
    entities_aabb[i] = {
      enemies.position[i - 1].x, enemies.position[i - 1].y,
      enemies.position[i - 1].x + enemies.size[i - 1].width,
      enemies.position[i - 1].y + enemies.size[i - 1].height, i};
  }

  std::array<AABB, kNumEntities> sorted_aabb = entities_aabb;
  std::sort(sorted_aabb.begin(), sorted_aabb.end(),
            [](const AABB& a, const AABB& b) { return a.min_x < b.min_x; });

  std::vector<CollisionPair> collision_pairs =
    GetCollisionPairsSAP(sorted_aabb);
  ResolveCollisionPairsSAP(player, enemies, entities_aabb, collision_pairs);
};

std::vector<CollisionPair> GetCollisionPairsSAP(
  std::array<AABB, kNumEntities> sorted_aabb) {
  std::vector<CollisionPair> collision_pairs;
  std::sort(sorted_aabb.begin(), sorted_aabb.end(),
            [](const AABB& a, const AABB& b) { return a.min_x < b.min_x; });
  std::vector<const AABB*> active_list;
  for (int i = 0; i < kNumEntities; ++i) {
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
          {current_aabb.entity_idx, active_aabb->entity_idx});
      };
    }

    // Add
    active_list.push_back(&current_aabb);
  }

  return collision_pairs;
};

void ResolveCollisionPairsSAP(Player& player, Enemies& enemies,
                              std::array<AABB, kNumEntities> entities_aabb,
                              std::vector<CollisionPair> collision_pairs) {
  for (const CollisionPair& cp : collision_pairs) {
    const AABB& a = entities_aabb[cp.index_a];
    const AABB& b = entities_aabb[cp.index_b];

    float overlap_x = std::min(a.max_x, b.max_x) - std::max(a.min_x, b.min_x);
    float overlap_y = std::min(a.max_y, b.max_y) - std::max(a.min_y, b.min_y);

    if (overlap_x <= 0.0f || overlap_y <= 0.0f)
      continue;  // No overlap

    // Choose smaller axis
    bool resolve_x = (overlap_x < overlap_y);

    auto MoveEntity = [&](int idx, float dx, float dy) {
      if (idx == 0) {
        player.position_.x += dx;
        player.position_.y += dy;
      } else {
        int enemy_idx = idx - 1;
        enemies.position[enemy_idx].x += dx;
        enemies.position[enemy_idx].y += dy;
      }
    };
    auto GetEntityCentroid = [&](int idx) -> Vector2D {
      if (idx == 0) {
        return rl2::GetCentroid(player.position_, player.stats_.size);
      } else {
        int enemy_idx = idx - 1;
        return rl2::GetCentroid(enemies.position[enemy_idx],
                                enemies.size[enemy_idx]);
      }
    };
    auto GetEntityInvMass = [&](int idx) -> float {
      if (idx == 0) {
        return player.stats_.inv_mass;
      } else {
        int enemy_idx = idx - 1;
        return enemies.inv_mass[enemy_idx];
      }
    };
    float inv_mass_a = GetEntityInvMass(cp.index_a);
    float inv_mass_b = GetEntityInvMass(cp.index_b);
    float push_factor = inv_mass_b / (inv_mass_a + inv_mass_b);

    if (resolve_x) {
      Vector2D centroid_a = GetEntityCentroid(cp.index_a);
      Vector2D centroid_b = GetEntityCentroid(cp.index_b);
      // Determine the separation direction: 1.0f if B is to the right of A, -1.0f otherwise
      float direction = (centroid_b.x - centroid_a.x >= 0.0f) ? 1.0f : -1.0f;

      MoveEntity(cp.index_a, -direction * overlap_x * (1.0f - push_factor),
                 0.0f);
      MoveEntity(cp.index_b, direction * overlap_x * push_factor, 0.0f);
    } else {
      Vector2D centroid_a = GetEntityCentroid(cp.index_a);
      Vector2D centroid_b = GetEntityCentroid(cp.index_b);
      // Determine the separation direction: 1.0f if B is to the right of A, -1.0f otherwise
      float direction = (centroid_b.y - centroid_a.y >= 0.0f) ? 1.0f : -1.0f;

      MoveEntity(cp.index_a, 0.0f,
                 -direction * overlap_y * (1.0f - push_factor));
      MoveEntity(cp.index_b, 0.0f, direction * overlap_y * push_factor);
    }
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
    if (projectiles.positions_[i].x < 0) {
      projectiles.DestroyProjectile(i);
    }
    if (projectiles.positions_[i].y < 0) {
      projectiles.DestroyProjectile(i);
    }
    if ((projectiles.positions_[i].x + projectiles.sizes_[i].width) >
        kMapWidth) {
      projectiles.DestroyProjectile(i);
    }
    if ((projectiles.positions_[i].y + projectiles.sizes_[i].height) >
        kMapHeight) {
      projectiles.DestroyProjectile(i);
    }
  }
};

}  // namespace rl2
