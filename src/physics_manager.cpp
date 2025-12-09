// src/physics_manager.cpp
#include "physics_manager.h"
#include "collision_manager.h"
#include "constants.h"
#include "entity.h"
#include "types.h"

namespace rl2 {
PhysicsManager::PhysicsManager(){};
PhysicsManager::~PhysicsManager(){};

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
    UpdateEnemyOccupancyMap(scene.enemy, scene.occupancy_map);
  }

  UpdateEnemyStatus(scene.enemy, scene.player);
  UpdateProjectilesStatus(scene.projectiles);

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

  auto GetGridTopLeft = [](Vector2D pos, Collider col) {
    Vector2D center = pos + col.offset;
    AABB collider_box = GetCollisionAABB(center, col.size);
    return WorldToGrid(Vector2D{collider_box.min_x, collider_box.min_y});
  };

  Vector2D player_grid_pos = GetGridTopLeft(player.position_, player.collider_);
  int player_grid_width = WorldToGrid(player.collider_.size.width);
  int player_grid_height = WorldToGrid(player.collider_.size.height);
  occupancy_map.SetGrid(static_cast<int>(player_grid_pos.x),
                        static_cast<int>(player_grid_pos.y), player_grid_width,
                        player_grid_height, player.entity_type_);

  for (int i = 0; i < kNumEnemies; ++i) {
    Vector2D enemygrid_pos =
        GetGridTopLeft(enemy.position[i], enemy.collider[i]);
    int enemy_grid_width = WorldToGrid(enemy.collider[i].size.width);
    int enemy_grid_height = WorldToGrid(enemy.collider[i].size.height);
    occupancy_map.SetGrid(static_cast<int>(enemygrid_pos.x),
                          static_cast<int>(enemygrid_pos.y), enemy_grid_width,
                          enemy_grid_height, enemy.entity_type);
  }

  const size_t num_proj = projectiles.GetNumProjectiles();
  for (int i = 0; i < num_proj; ++i) {
    Vector2D proj_grid_pos =
        GetGridTopLeft(projectiles.position_[i], projectiles.collider_[i]);
    int proj_grid_width = WorldToGrid(projectiles.collider_[i].size.width);
    int proj_grid_height = WorldToGrid(projectiles.collider_[i].size.height);
    occupancy_map.SetGrid(static_cast<int>(proj_grid_pos.x),
                          static_cast<int>(proj_grid_pos.y), proj_grid_width,
                          proj_grid_height, projectiles.entity_type_);
  };
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

    if (enemy.timeout_timer[i] >= kEpisodeTimeout) {
      enemy.is_done[i] = true;
      enemy.is_truncated_latched[i] = true;
    }
  };
};

void PhysicsManager::UpdateProjectilesStatus(Projectiles& projectiles) {
  projectiles.DestroyProjectiles();
};

}  // namespace rl2
