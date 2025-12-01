// src/game_init.cpp
#include <algorithm>
#include <csignal>
#include "game.h"
#include "random.h"

namespace rl2 {

bool Game::Initialize() {

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGKILL, SignalHandler);
  game_status_.is_debug = true;
  game_status_.is_headless = false;

  if (!(renderer_.Initialize(game_status_.is_headless))) {
    return false;
  }

  if (!(physics_.Initialize())) {
    return false;
  }

  if (!(Game::InitializePlayer())) {
    return false;
  }
  if (!(Game::InitializeEnemies())) {
    return false;
  }
  if (!(Game::InitializeCamera())) {
    return false;
  }

  time_ = (float)(SDL_GetTicks64() / 1000.0f);
  game_state_ = GameState::is_running;

  return true;
}

bool Game::InitializePlayer() {
  player_.stats_.size = {kPlayerWidth, kPlayerHeight};
  player_.stats_.inv_mass = kPlayerInvMass;
  player_.position_ = {kPlayerInitX, kPlayerInitY};
  player_.stats_.movement_speed = kPlayerSpeed;
  player_.UpdateAllSpellStats();
  return true;
};

bool Game::InitializeEnemies() {
  std::fill(enemy_.is_alive.begin(), enemy_.is_alive.end(), false);
  std::fill(enemy_.movement_speed.begin(), enemy_.movement_speed.end(),
            kEnemySpeed);
  std::fill(enemy_.size.begin(), enemy_.size.end(),
            Size{kEnemyHeight, kEnemyWidth});
  std::fill(enemy_.inv_mass.begin(), enemy_.inv_mass.end(), kEnemyInvMass);
  RespawnEnemy(enemy_, player_);

  // Add slight variation to each enemy to make it more interesting.
  for (int i = 0; i < kNumEnemies; ++i) {
    enemy_.movement_speed[i] += GenerateRandomInt(1, 100);
    enemy_.size[i].height += GenerateRandomInt(1, 50);
    enemy_.size[i].width += GenerateRandomInt(1, 50);
  };

  return true;
};

bool Game::InitializeCamera() {
  Vector2D player_centroid =
      GetCentroid(player_.position_, player_.stats_.size);
  renderer_.camera_.position_.x = player_centroid.x - 0.5f * kWindowWidth;
  renderer_.camera_.position_.y = player_centroid.y - 0.5f * kWindowHeight;

  return true;
};

}  // namespace rl2
