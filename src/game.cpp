// src/game.cpp
#include "game.h"
#include <SDL2/SDL_timer.h>
#include <SDL_render.h>
#include <SDL_surface.h>
#include <array>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <vector>
#include "constants.h"
#include "entity.h"
#include "physics_manager.h"
#include "render_manager.h"
#include "types.h"

namespace rl2 {

volatile std::sig_atomic_t Game::stop_request_ = 0;

void Game::SignalHandler(int signal) {
  stop_request_ = 1;
}

Game::Game(){};

Game::~Game() {
  renderer_.Shutdown();
}

int Game::GetGameState() {
  return static_cast<int>(game_state_);
};

void FrameStats::update_frame_time_buffer(float new_value) {
  float oldest_value = frame_time_buffer[head_index];
  frame_time_sum = frame_time_sum - oldest_value + new_value;
  frame_time_buffer[head_index] = new_value;

  if (current_buffer_length < max_buffer_length) {
    current_buffer_length++;
  };

  head_index = (head_index + 1) % max_buffer_length;
};

float FrameStats::get_average_frame_time() {
  if (current_buffer_length == 0) {
    return 0.0f;
  }
  return static_cast<float>(frame_time_sum / current_buffer_length);
};

void FrameStats::print_fps_running_average(float dt) {
  static float accumulated_time = 0.0f;
  update_frame_time_buffer(dt);
  if (accumulated_time > 1.0f) {
    float avg_frame_time = get_average_frame_time();
    float average_fps = 1.0f / (avg_frame_time);
    std::cout << "Current Avg FPS: " << std::fixed << average_fps;
    std::cout << "\r" << std::flush;
    accumulated_time -= 1.0f;
  };
  accumulated_time += dt;
};

void Game::StepGame() {
  CachePreviousState();
  ProcessInput();
  physics_.StepPhysics(scene_);
  time_ += physics_.GetPhysicsDt();
};

void Game::RenderGame(float alpha) {
  renderer_.Render(scene_, alpha, game_status_.is_debug);
};

void Game::RunGameLoop() {
  // Runs the game loop continuously.

  float current_time = static_cast<float>(SDL_GetTicks64() / 1000.0f);
  float accumulator = 0.0f;

  while (game_state_ == GameState::is_running) {
    float new_time = (float)(SDL_GetTicks64() / 1000.0f);
    float frame_time = new_time - current_time;
    current_time = new_time;

    // In case the frame time is too large, we override the frame time and
    // use a specified max frame time instead to avoid the "spiral of death".
    if (frame_time > kMaxFrameTime) {
      frame_time = kMaxFrameTime;
    }

    accumulator += frame_time;

    while (accumulator >= physics_.GetPhysicsDt()) {
      StepGame();
      accumulator -= physics_.GetPhysicsDt();
    }

    float alpha = accumulator / dt;

    if (game_status_.is_headless) {
      return;
    }
    RenderGame(alpha);
    game_status_.frame_stats.print_fps_running_average(frame_time);
  }
};

void Game::ProcessInput() {

  // To be able to quit while in headless mode we need to capture ctrl+C signals
  if (stop_request_) {
    game_state_ = GameState::in_shutdown;
    std::cout << "Signal received. Exiting..." << std::endl;
    return;
  };

  if (game_status_.is_headless) {
    return;
  }

  SDL_Event e;

  while (SDL_PollEvent(&e) != 0) {
    if (e.type == SDL_QUIT) {
      game_state_ = GameState::in_shutdown;
    } else if (e.type == SDL_KEYDOWN) {

      switch (e.key.keysym.sym) {
        case SDLK_q:
          game_state_ = GameState::in_shutdown;
          std::cout << "Key 'q' pressed! Exiting..." << std::endl;
          break;
      }
    }
  }
  cursor_position_ = GetCursorPositionWorld();
  Game::ProcessPlayerInput();
}

Vector2D Game::GetCursorPositionWorld() {
  int cursor_x, cursor_y;
  uint32_t cursor_mask = SDL_GetMouseState(&cursor_x, &cursor_y);

  return {(float)(cursor_x + renderer_.camera_.position_.x),
          (float)(cursor_y + renderer_.camera_.position_.y)};
};

void Game::ProcessPlayerInput() {
  scene_.player.velocity_ = {0.0f, 0.0f};
  const Uint8* currentKeyStates = SDL_GetKeyboardState(NULL);
  if (currentKeyStates[SDL_SCANCODE_W]) {
    scene_.player.velocity_.y -= 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_S]) {
    scene_.player.velocity_.y += 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_A]) {
    scene_.player.velocity_.x -= 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_D]) {
    scene_.player.velocity_.x += 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_F]) {
    std::optional<ProjectileData> fireball = scene_.player.CastProjectileSpell(
        scene_.player.fireball_, time_, cursor_position_);

    if (fireball.has_value()) {
      scene_.projectiles.AddProjectile(*fireball);
    }
  }
  if (currentKeyStates[SDL_SCANCODE_I]) {
    std::optional<ProjectileData> frostbolt = scene_.player.CastProjectileSpell(
        scene_.player.frostbolt_, time_, cursor_position_);

    if (frostbolt.has_value()) {
      scene_.projectiles.AddProjectile(*frostbolt);
    }
  }
}

void Game::CachePreviousState() {
  scene_.player.prev_position_ = scene_.player.position_;

  for (int i = 0; i < kNumEnemies; ++i) {
    if (scene_.enemy.is_alive[i]) {
      scene_.enemy.prev_position[i] = scene_.enemy.position[i];
    }
  }

  size_t num_proj = scene_.projectiles.GetNumProjectiles();
  for (size_t i = 0; i < num_proj; ++i) {
    scene_.projectiles.prev_position_[i] = scene_.projectiles.position_[i];
  }

  renderer_.camera_.prev_position_ = renderer_.camera_.position_;
}

void Game::Shutdown() {

  renderer_.Shutdown();
}

}  // namespace rl2
