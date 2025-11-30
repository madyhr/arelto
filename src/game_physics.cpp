// src/game_physics.cpp
#include "collision.h"
#include "game.h"

namespace rl2 {
void rl2::Game::StepPhysics(float physics_dt) {

  Game::UpdatePlayerPosition(physics_dt);
  Game::UpdateEnemyPosition(physics_dt);
  Game::UpdateProjectilePosition(physics_dt);
  Game::HandleCollisions();
  Game::HandleOutOfBounds();

  UpdateEnemyStatus(enemy_, player_);
  Game::UpdateWorldOccupancyMap();
  Game::UpdateEnemyOccupancyMap();
  projectiles_.DestroyProjectiles();
}

void Game::UpdatePlayerPosition(float dt) {
  Vector2D delta_pos =
      player_.velocity_.Normalized() * (player_.stats_.movement_speed * dt);
  player_.position_ += delta_pos;
};

void Game::UpdateEnemyPosition(float dt) {
  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemy_.is_alive[i]) {
      Vector2D distance_vector = player_.position_ - enemy_.position[i];
      enemy_.velocity[i] = distance_vector.Normalized();
      enemy_.position[i] += enemy_.velocity[i] * enemy_.movement_speed[i] * dt;
    }
  }
};

void Game::UpdateProjectilePosition(float dt) {
  size_t num_projectiles = projectiles_.GetNumProjectiles();
  if (num_projectiles == 0) {
    return;
  };

  for (int i = 0; i < num_projectiles; ++i) {
    projectiles_.position_[i] +=
        projectiles_.direction_[i] * projectiles_.speed_[i] * dt;
  };
};

void Game::HandleCollisions() {
  rl2::HandleCollisionsSAP(player_, enemy_, projectiles_);
};

void Game::HandleOutOfBounds() {
  HandlePlayerOOB(player_);
  HandleEnemyOOB(enemy_);
  HandleProjectileOOB(projectiles_);
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

void Game::UpdateWorldOccupancyMap() {

  world_occupancy_map_.Clear();

  Vector2D player_grid_pos = WorldToGrid(player_.position_);
  int player_grid_width = WorldToGrid(player_.stats_.size.width);
  int player_grid_height = WorldToGrid(player_.stats_.size.height);
  // Vector2D player_grid_size =
  //     WorldToGrid({player_.stats_.size.width, player_.stats_.size.height});
  world_occupancy_map_.SetGrid(
      static_cast<int>(player_grid_pos.x), static_cast<int>(player_grid_pos.y),
      player_grid_width, player_grid_height, player_.entity_type_);
  for (int i = 0; i < kNumEnemies; ++i) {
    Vector2D enemy_grid_pos = WorldToGrid(enemy_.position[i]);
    int enemy_grid_width = WorldToGrid(enemy_.size[i].width);
    int enemy_grid_height = WorldToGrid(enemy_.size[i].height);
    world_occupancy_map_.SetGrid(
        static_cast<int>(enemy_grid_pos.x), static_cast<int>(enemy_grid_pos.y),
        enemy_grid_width, enemy_grid_height, enemy_.entity_type);
  }

  const size_t num_proj = projectiles_.GetNumProjectiles();
  for (int i = 0; i < num_proj; ++i) {
    Vector2D proj_grid_pos = WorldToGrid(projectiles_.position_[i]);
    int proj_grid_width = WorldToGrid(projectiles_.size_[i].width);
    int proj_grid_height = WorldToGrid(projectiles_.size_[i].height);
    world_occupancy_map_.SetGrid(
        static_cast<int>(proj_grid_pos.x), static_cast<int>(proj_grid_pos.y),
        proj_grid_width, proj_grid_height, projectiles_.entity_type_);
  };
};

void Game::UpdateEnemyOccupancyMap() {
  // Get local occupancy map from world
  const int local_h = kEnemyOccupancyMapHeight;
  const int half_w = static_cast<int>(kEnemyOccupancyMapWidth / 2);
  const int half_h = static_cast<int>(kEnemyOccupancyMapHeight / 2);

  for (int i = 0; i < kNumEnemies; ++i) {
    // We do not update the occupancy map for enemies that are dead to save performance.
    if (!enemy_.is_alive[i]) {
      continue;
    }
    // Vector2D enemy_grid_pos = WorldToGrid(enemy_.position[i]);
    Vector2D enemy_grid_pos =
        WorldToGrid(GetCentroid(enemy_.position[i], enemy_.size[i]));

    int start_world_x = static_cast<int>(enemy_grid_pos.x) - half_w;
    int start_world_y = static_cast<int>(enemy_grid_pos.y) - half_h;

    for (int local_row = 0; local_row < local_h; ++local_row) {
      int world_row = start_world_y + local_row;
      enemy_.occupancy_map[i].CopyRowFrom(world_occupancy_map_, local_row,
                                          world_row, start_world_x);
    };
  }
};

}  // namespace rl2
