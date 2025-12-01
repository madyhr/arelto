// src/physics_manager.cpp
#include "physics_manager.h"
#include "collision.h"
#include "constants.h"
#include "entity.h"
#include "game.h"

namespace rl2 {
PhysicsManager::PhysicsManager(){};
PhysicsManager::~PhysicsManager(){};

bool PhysicsManager::Initialize() {
  SetPhysicsDt(kPhysicsDt);

  return true;
};

void PhysicsManager::StepPhysics(
    Player& player, Enemy& enemy, Projectiles& projectiles,
    FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map) {

  UpdatePlayerState(player);
  UpdateEnemyState(enemy, player);
  UpdateProjectileState(projectiles);
  HandleCollisions(player, enemy, projectiles);
  HandleOutOfBounds(player, enemy, projectiles);

  UpdateEnemyStatus(enemy, player);
  UpdateProjectilesStatus(projectiles);
  UpdateWorldOccupancyMap(occupancy_map, player, enemy, projectiles);
  UpdateEnemyOccupancyMap(enemy, occupancy_map);

  projectiles.DestroyProjectiles();
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
      Vector2D distance_vector = player.position_ - enemy.position[i];
      enemy.velocity[i] = distance_vector.Normalized();
      enemy.position[i] +=
          enemy.velocity[i] * enemy.movement_speed[i] * physics_dt_;

      // Update last horizontal velocity for assets that depend on the direction
      // an enemy is facing.
      if (enemy.velocity[i].x != 0) {
        enemy.last_horizontal_velocity[i] = enemy.velocity[i].x;
      }
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

void PhysicsManager::HandleCollisions(Player& player, Enemy& enemy,
                                      Projectiles& projectiles) {
  rl2::HandleCollisionsSAP(player, enemy, projectiles);
};

void PhysicsManager::HandleOutOfBounds(Player& player, Enemy& enemy,
                                       Projectiles& projectiles) {
  HandlePlayerOOB(player);
  HandleEnemyOOB(enemy);
  HandleProjectileOOB(projectiles);
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

void HandleEnemyOOB(Enemy& enemies) {
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

void PhysicsManager::UpdateWorldOccupancyMap(
    FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map,
    Player& player, Enemy& enemy, Projectiles& projectiles) {

  occupancy_map.Clear();

  Vector2D player_grid_pos = WorldToGrid(player.position_);
  int player_grid_width = WorldToGrid(player.stats_.size.width);
  int player_grid_height = WorldToGrid(player.stats_.size.height);
  occupancy_map.SetGrid(static_cast<int>(player_grid_pos.x),
                        static_cast<int>(player_grid_pos.y), player_grid_width,
                        player_grid_height, player.entity_type_);

  for (int i = 0; i < kNumEnemies; ++i) {
    Vector2D enemygrid_pos = WorldToGrid(enemy.position[i]);
    int enemy_grid_width = WorldToGrid(enemy.size[i].width);
    int enemy_grid_height = WorldToGrid(enemy.size[i].height);
    occupancy_map.SetGrid(static_cast<int>(enemygrid_pos.x),
                          static_cast<int>(enemygrid_pos.y), enemy_grid_width,
                          enemy_grid_height, enemy.entity_type);
  }

  const size_t num_proj = projectiles.GetNumProjectiles();
  for (int i = 0; i < num_proj; ++i) {
    Vector2D proj_grid_pos = WorldToGrid(projectiles.position_[i]);
    int proj_grid_width = WorldToGrid(projectiles.size_[i].width);
    int proj_grid_height = WorldToGrid(projectiles.size_[i].height);
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
        WorldToGrid(GetCentroid(enemy.position[i], enemy.size[i]));

    int start_world_x = static_cast<int>(enemy_grid_pos.x) - half_w;
    int start_world_y = static_cast<int>(enemy_grid_pos.y) - half_h;

    for (int local_row = 0; local_row < local_h; ++local_row) {
      int world_row = start_world_y + local_row;
      enemy.occupancy_map[i].CopyRowFrom(occupancy_map, local_row, world_row,
                                         start_world_x);
    };
  }
};

}  // namespace rl2
