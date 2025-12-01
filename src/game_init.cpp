// src/game_init.cpp
#include <csignal>
#include "game.h"

namespace rl2 {

bool Game::Initialize() {

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGKILL, SignalHandler);
  game_status_.is_debug = false;
  game_status_.is_headless = false;

  if (!(render_manager_.Initialize(game_status_.is_headless))) {
    return false;
  }

  if (!(physics_manager_.Initialize())) {
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
      GetCentroid(scene_.player.position_, scene_.player.stats_.size);
  render_manager_.camera_.position_.x = player_centroid.x - 0.5f * kWindowWidth;
  render_manager_.camera_.position_.y =
      player_centroid.y - 0.5f * kWindowHeight;

  return true;
};

}  // namespace rl2
