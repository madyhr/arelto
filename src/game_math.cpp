// src/game_math.cpp
#include "game_math.h"
#include <SDL_rect.h>
#include <array>
#include <cmath>
#include <vector>
#include "constants.h"
#include "entity.h"
#include "types.h"

namespace rl2 {

float CalculateVector2DDistance(Vector2D v0, Vector2D v1) {
  float dx = v1.x - v0.x;
  float dy = v1.y - v0.y;
  float distance = std::hypot(dx, dy);
  return distance;
};

Vector2D GetCentroid(Vector2D position, Size size) {
  return {position.x + 0.5f * size.width, position.y + 0.5f * size.height};
}

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

}  // namespace rl2
