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

  scene_.Reset();

  if (!(Game::InitializeCamera())) {
    return false;
  }

  time_ = (float)(SDL_GetTicks64() / 1000.0f);
  game_state_ = GameState::is_running;

  return true;
}

bool Game::InitializeCamera() {
  Vector2D player_centroid =
      GetCentroid(player_.position_, player_.stats_.size);
  renderer_.camera_.position_.x = player_centroid.x - 0.5f * kWindowWidth;
  renderer_.camera_.position_.y = player_centroid.y - 0.5f * kWindowHeight;

  return true;
};

}  // namespace rl2
