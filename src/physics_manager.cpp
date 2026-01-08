// src/physics_manager.cpp
#include "physics_manager.h"
#include <algorithm>
#include "collision_manager.h"
#include "constants/enemy.h"
#include "constants/game.h"
#include "constants/map.h"
#include "constants/ray_caster.h"
#include "entity.h"
#include "types.h"

namespace rl2 {
PhysicsManager::PhysicsManager() {};
PhysicsManager::~PhysicsManager() {};

bool PhysicsManager::Initialize() {
  SetPhysicsDt(kPhysicsDt);

  return true;
};

void PhysicsManager::StepPhysics(Scene& scene) {

  UpdatePlayerState(scene.player);
  UpdateEnemyState(scene.enemy, scene.player);
  UpdateProjectileState(scene.projectiles);

  HandleCollisions(scene);
  HandleOutOfBounds(scene.player, scene.enemy, scene.projectiles);

  if (tick_count_ % kOccupancyMapTimeDecimation == 0) {
    UpdateWorldOccupancyMap(scene.occupancy_map, scene.player, scene.enemy,
                            scene.projectiles);
    // UpdateEnemyOccupancyMap(scene.enemy, scene.occupancy_map);
    UpdateEnemyRayCaster(scene.enemy, scene.occupancy_map);
  }

  UpdateEnemyStatus(scene.enemy, scene.player);
  UpdatePlayerStatus(scene.player, scene.enemy);
  UpdateProjectilesStatus(scene.projectiles);
  UpdateGemStatus(scene.exp_gem);

  tick_count_ += 1;
}

void PhysicsManager::UpdatePlayerState(Player& player) {
  Vector2D delta_pos = player.velocity_.Normalized() *
                       (player.stats_.movement_speed * physics_dt_);
  player.position_ += delta_pos;

  if (player.velocity_.x != 0) {
    player.last_horizontal_velocity_ = player.velocity_.x;
  }
};

void PhysicsManager::UpdateEnemyState(Enemy& enemy, const Player& player) {
  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemy.is_alive[i]) {
      enemy.velocity[i] = enemy.velocity[i].Normalized();
      enemy.position[i] +=
          enemy.velocity[i] * enemy.movement_speed[i] * physics_dt_;

      // Update last horizontal velocity for assets that depend on the direction
      // an enemy is facing.
      if (enemy.velocity[i].x != 0) {
        enemy.last_horizontal_velocity[i] = enemy.velocity[i].x;
      }

      if (enemy.attack_cooldown[i] >= 0.0f) {
        enemy.attack_cooldown[i] -= physics_dt_;
      }

      enemy.damage_dealt_sim_step[i] = 0;
    }
  }
};

void PhysicsManager::UpdateProjectileState(Projectiles& projectiles) {
  size_t num_projectiles = projectiles.GetNumProjectiles();
  if (num_projectiles == 0) {
    return;
  };

  for (int i = 0; i < num_projectiles; ++i) {
    projectiles.position_[i] +=
        projectiles.direction_[i] * projectiles.speed_[i] * physics_dt_;
  };
};

void PhysicsManager::HandleCollisions(Scene& scene) {
  collision_manager_.HandleCollisionsSAP(scene);
};

void PhysicsManager::HandleOutOfBounds(Player& player, Enemy& enemy,
                                       Projectiles& projectiles) {
  HandlePlayerOOB(player);
  HandleEnemyOOB(enemy);
  HandleProjectileOOB(projectiles);
};

void PhysicsManager::HandlePlayerOOB(Player& player) {
  if (player.position_.x < 0) {
    player.position_.x = 0;
  }
  if (player.position_.y < 0) {
    player.position_.y = 0;
  }
  if ((player.position_.x + player.stats_.sprite_size.width) > kMapWidth) {
    player.position_.x = kMapWidth - player.stats_.sprite_size.width;
  }
  if ((player.position_.y + player.stats_.sprite_size.height) > kMapHeight) {
    player.position_.y = kMapHeight - player.stats_.sprite_size.height;
  }
};

void PhysicsManager::HandleEnemyOOB(Enemy& enemies) {
  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemies.is_alive[i]) {
      if (enemies.position[i].x < 0) {
        enemies.position[i].x = 0;
      }
      if (enemies.position[i].y < 0) {
        enemies.position[i].y = 0;
      }
      if ((enemies.position[i].x + enemies.sprite_size[i].width) > kMapWidth) {
        enemies.position[i].x = kMapWidth - enemies.sprite_size[i].width;
      }
      if ((enemies.position[i].y + enemies.sprite_size[i].height) >
          kMapHeight) {
        enemies.position[i].y = kMapHeight - enemies.sprite_size[i].height;
      }
    }
  }
};

void PhysicsManager::HandleProjectileOOB(Projectiles& projectiles) {
  size_t num_projectiles = projectiles.GetNumProjectiles();
  if (num_projectiles == 0) {
    return;
  }
  for (int i = 0; i < num_projectiles; ++i) {
    if (projectiles.position_[i].x < 0 || projectiles.position_[i].y < 0 ||
        (projectiles.position_[i].x + projectiles.sprite_size_[i].width) >
            kMapWidth ||
        (projectiles.position_[i].y + projectiles.sprite_size_[i].height) >
            kMapHeight) {
      projectiles.to_be_destroyed_.insert(i);
    }
  }
};

void PhysicsManager::UpdateWorldOccupancyMap(
    FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map,
    Player& player, Enemy& enemy, Projectiles& projectiles) {

  occupancy_map.Clear();

  // Helper lambda to use an entity's AABB to set the entity type on the
  // world occupancy map accordingly.
  auto MarkOccupancy = [&](Vector2D pos, Collider col, EntityType type) {
    Vector2D center = pos + col.offset;
    AABB aabb = GetCollisionAABB(center, col.size);

    Vector2D grid_min = WorldToGrid(Vector2D{aabb.min_x, aabb.min_y});
    Vector2D grid_max = WorldToGrid(Vector2D{aabb.max_x, aabb.max_y});

    int start_x = static_cast<int>(grid_min.x);
    int start_y = static_cast<int>(grid_min.y);
    int end_x = static_cast<int>(grid_max.x);
    int end_y = static_cast<int>(grid_max.y);

    // Clamp to map bounds (just for safety)
    start_x = std::max(0, start_x);
    start_y = std::max(0, start_y);
    end_x = std::min(kOccupancyMapWidth - 1, end_x);
    end_y = std::min(kOccupancyMapHeight - 1, end_y);

    for (int x = start_x; x <= end_x; ++x) {
      for (int y = start_y; y <= end_y; ++y) {
        occupancy_map.Set(x, y, type);
      }
    }
  };

  MarkOccupancy(player.position_, player.collider_, player.entity_type_);

  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemy.is_alive[i]) {
      MarkOccupancy(enemy.position[i], enemy.collider[i], enemy.entity_type);
    }
  }

  const size_t num_proj = projectiles.GetNumProjectiles();
  for (int i = 0; i < num_proj; ++i) {
    MarkOccupancy(projectiles.position_[i], projectiles.collider_[i],
                  projectiles.entity_type_);
  };

  occupancy_map.AddBorder(EntityType::terrain);
};

void PhysicsManager::UpdateEnemyOccupancyMap(
    Enemy& enemy,
    FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map) {
  // Get local occupancy map from world
  const int local_h = kEnemyOccupancyMapHeight;
  const int half_w = static_cast<int>(kEnemyOccupancyMapWidth / 2);
  const int half_h = static_cast<int>(kEnemyOccupancyMapHeight / 2);

  for (int i = 0; i < kNumEnemies; ++i) {
    // We do not update the occupancy map for enemies that are dead to save performance.
    if (!enemy.is_alive[i]) {
      continue;
    }
    Vector2D enemy_grid_pos =
        WorldToGrid(GetCentroid(enemy.position[i], enemy.sprite_size[i]));

    int start_world_x = static_cast<int>(enemy_grid_pos.x) - half_w;
    int start_world_y = static_cast<int>(enemy_grid_pos.y) - half_h;

    for (int local_row = 0; local_row < local_h; ++local_row) {
      int world_row = start_world_y + local_row;
      enemy.occupancy_map[i].CopyRowFrom(occupancy_map, local_row, world_row,
                                         start_world_x);
    };
  }
};

void PhysicsManager::UpdateEnemyStatus(Enemy& enemy, const Player& player) {

  for (int i = 0; i < kNumEnemies; ++i) {
    enemy.timeout_timer[i] += physics_dt_;

    if (enemy.health_points[i] <= 0) {
      enemy.is_alive[i] = false;
      enemy.is_done[i] = true;
      enemy.is_terminated_latched[i] = true;
    };
  };
};

void PhysicsManager::UpdatePlayerStatus(Player& player, const Enemy& enemy) {

  while (player.stats_.exp_points >= player.stats_.exp_points_required) {
    player.stats_.level++;
    player.stats_.exp_points -= player.stats_.exp_points_required;
    player.stats_.exp_points_required = static_cast<int>(
        kPlayerExpRequiredScale * player.stats_.exp_points_required);
  }
};

void PhysicsManager::UpdateEnemyRayCaster(
    Enemy& enemy,
    const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map) {

  int history_idx = enemy.ray_caster.history_idx;

  for (int ray_idx = 0; ray_idx < kNumRays; ++ray_idx) {
    Vector2D dir = enemy.ray_caster.pattern.ray_dir[ray_idx];

    for (int enemy_idx = 0; enemy_idx < kNumEnemies; ++enemy_idx) {
      if (!enemy.is_alive[enemy_idx]) {
        continue;
      }

      Vector2D center =
          enemy.position[enemy_idx] + enemy.collider[enemy_idx].offset;

      float half_w = enemy.collider[enemy_idx].size.width * 0.5f;
      float half_h = enemy.collider[enemy_idx].size.height * 0.5f;

      // a small offset is added to ensure the rays do not clip the corners
      // of the collider. Note: this does add blind spots.
      float ray_offset = std::max(half_h, half_w) + kMinRayDistance;
      Vector2D start_pos = center + dir * ray_offset;

      Vector2D grid_pos = WorldToGrid(start_pos);
      EntityType start_cell_type = occupancy_map.Get(
          static_cast<int>(grid_pos.x), static_cast<int>(grid_pos.y));

      // Before actually casting a ray, since we cast the ray offset from the
      // center position, we need to check if we are about to cast through
      // terrain which could lead to a ray going OOB. In that case, we should
      // just skip the ray casting altogether and we can set the distance to 0.
      bool out_of_bounds = grid_pos.x >= kOccupancyMapWidth - 1 ||
                           grid_pos.y >= kOccupancyMapHeight - 1 ||
                           grid_pos.x <= 1 || grid_pos.y <= 1;
      RayHit ray_hit;
      if (out_of_bounds) {
        ray_hit = {0.0f, EntityType::terrain};
      } else {
        ray_hit = CastRay(start_pos, dir, occupancy_map);
      }

      enemy.ray_caster.ray_hit_distances[history_idx][ray_idx][enemy_idx] =
          ray_hit.distance;
      enemy.ray_caster.ray_hit_types[history_idx][ray_idx][enemy_idx] =
          ray_hit.entity_type;
    }
  }

  enemy.ray_caster.history_idx =
      (enemy.ray_caster.history_idx + 1) % kRayHistoryLength;
};

void PhysicsManager::UpdateProjectilesStatus(Projectiles& projectiles) {
  projectiles.DestroyProjectiles();
};

void PhysicsManager::UpdateGemStatus(ExpGem& exp_gem) {
  exp_gem.DestroyExpGems();
};

}  // namespace rl2
