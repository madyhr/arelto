// src/physics_manager.cpp
#include "physics_manager.h"
#include "collision_manager.h"
#include "constants/enemy.h"
#include "constants/game.h"
#include "constants/map.h"
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

}  // namespace rl2
