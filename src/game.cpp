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
#include "types.h"

namespace rl2 {

volatile std::sig_atomic_t Game::stop_request_ = 0;

void Game::SignalHandler(int signal) {
  stop_request_ = 1;
}

Game::Game(){};

Game::~Game() {
  Game::Shutdown();
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

void Game::Step() {
  CachePreviousState();
  ProcessInput();
  StepPhysics(dt);
  time_ += dt;
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

    while (accumulator >= dt) {
      CachePreviousState();
      ProcessInput();
      Game::StepPhysics(dt);
      accumulator -= dt;
      time_ += dt;
    }

    float alpha = accumulator / dt;

    if (game_status_.in_headless_mode) {
      return;
    }
    Game::GenerateOutput(alpha);
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

  if (game_status_.in_headless_mode) {
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

  return {(float)(cursor_x + camera_.position_.x),
          (float)(cursor_y + camera_.position_.y)};
};

void Game::ProcessPlayerInput() {
  player_.velocity_ = {0.0f, 0.0f};
  const Uint8* currentKeyStates = SDL_GetKeyboardState(NULL);
  if (currentKeyStates[SDL_SCANCODE_W]) {
    player_.velocity_.y -= 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_S]) {
    player_.velocity_.y += 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_A]) {
    player_.velocity_.x -= 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_D]) {
    player_.velocity_.x += 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_F]) {
    std::optional<ProjectileData> fireball =
        player_.CastProjectileSpell(player_.fireball_, time_, cursor_position_);

    if (fireball.has_value()) {
      projectiles_.AddProjectile(*fireball);
    }
  }
  if (currentKeyStates[SDL_SCANCODE_I]) {
    std::optional<ProjectileData> frostbolt = player_.CastProjectileSpell(
        player_.frostbolt_, time_, cursor_position_);

    if (frostbolt.has_value()) {
      projectiles_.AddProjectile(*frostbolt);
    }
  }
}

void Game::CachePreviousState() {
  player_.prev_position_ = player_.position_;

  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemy_.is_alive[i]) {
      enemy_.prev_position[i] = enemy_.position[i];
    }
  }

  size_t num_proj = projectiles_.GetNumProjectiles();
  for (size_t i = 0; i < num_proj; ++i) {
    projectiles_.prev_position_[i] = projectiles_.position_[i];
  }

  camera_.prev_position_ = camera_.position_;
}

void Game::Shutdown() {

  if (resources_.player_texture) {
    SDL_DestroyTexture(resources_.player_texture);
    resources_.player_texture = nullptr;
  }

  if (resources_.enemy_texture) {
    SDL_DestroyTexture(resources_.enemy_texture);
    resources_.enemy_texture = nullptr;
  }

  IMG_Quit();

  if (resources_.renderer) {
    SDL_DestroyRenderer(resources_.renderer);
    resources_.renderer = nullptr;
  }

  if (resources_.window) {
    SDL_DestroyWindow(resources_.window);
    resources_.window = nullptr;
  }

  SDL_Quit();
}

}  // namespace rl2
