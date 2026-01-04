// src/game.cpp
#include "game.h"
#include <SDL2/SDL_timer.h>
#include <SDL_mouse.h>
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

Game::Game() {};
Game::~Game() {}

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

  if (!(reward_manager_.Initialize())) {
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
      GetCentroid(scene_.player.position_, scene_.player.stats_.sprite_size);
  render_manager_.camera_.position_.x = player_centroid.x - 0.5f * kWindowWidth;
  render_manager_.camera_.position_.y =
      player_centroid.y - 0.5f * kWindowHeight;

  return true;
};

int Game::GetGameState() {
  return static_cast<int>(game_state_);
};

void Game::StepGamePhysics() {
  physics_manager_.StepPhysics(scene_);
  reward_manager_.UpdateRewardTerms(scene_);
  time_ += physics_manager_.GetPhysicsDt();

  if (scene_.player.stats_.health <= 0) {
    game_state_ = is_gameover;
  };
  CachePreviousState();
};

void Game::RenderGame(float alpha) {
  render_manager_.Render(scene_, alpha, game_status_.is_debug, time_,
                         game_state_);
};

void Game::ResetGame() {
  scene_.Reset();
  time_ = 0.0f;
  accumulator_step_ = 0.0f;
  game_state_ = is_running;
};

void Game::RunGameLoop() {
  // Runs the game loop continuously.
  float current_time = static_cast<float>(SDL_GetTicks64() / 1000.0f);
  float accumulator = 0.0f;

  while (game_state_ != in_shutdown) {

    ProcessInput();

    switch (game_state_) {

      case is_running: {
        float new_time = (float)(SDL_GetTicks64() / 1000.0f);
        float frame_time = new_time - current_time;
        current_time = new_time;

        // In case the frame time is too large, we override the frame time and
        // use a specified max frame time instead to avoid the "spiral of death".
        if (frame_time > kMaxFrameTime) {
          frame_time = kMaxFrameTime;
        }

        accumulator += frame_time;

        while (accumulator >= physics_manager_.GetPhysicsDt()) {
          StepGamePhysics();
          accumulator -= physics_manager_.GetPhysicsDt();
        }

        float alpha = accumulator / physics_manager_.GetPhysicsDt();

        if (game_status_.is_headless) {
          break;
        }

        RespawnEnemy(scene_.enemy, scene_.player);
        RenderGame(alpha);
        game_status_.frame_stats.print_fps_running_average(frame_time);
        break;
      }

      case in_start_screen:
        break;

      case is_gameover:
        break;

      default:
        break;
    }
  }
};

void Game::StepGame(float dt) {
  accumulator_step_ += dt;

  while (accumulator_step_ >= physics_manager_.GetPhysicsDt()) {
    StepGamePhysics();
    accumulator_step_ -= physics_manager_.GetPhysicsDt();
  }

  RespawnEnemy(scene_.enemy, scene_.player);
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

        case SDLK_r:
          if (game_state_ == is_gameover) {
            ResetGame();
          }
      }
    }
  }

  int cursor_pos_x, cursor_pos_y;
  uint32_t mouse_state = SDL_GetMouseState(&cursor_pos_x, &cursor_pos_y);

  cursor_position_ = {
      (float)(cursor_pos_x + render_manager_.camera_.position_.x),
      (float)(cursor_pos_y + render_manager_.camera_.position_.y)};

  Game::ProcessPlayerInput(mouse_state);
}

void Game::ProcessPlayerInput(uint32_t mouse_state) {
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
  if (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
    std::optional<ProjectileData> fireball = scene_.player.CastProjectileSpell(
        scene_.player.fireball_, time_, cursor_position_);

    if (fireball.has_value()) {
      scene_.projectiles.AddProjectile(*fireball);
    }
  }
  if (mouse_state & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
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
      scene_.enemy.prev_velocity[i] = scene_.enemy.velocity[i];
    }
  }

  size_t num_proj = scene_.projectiles.GetNumProjectiles();
  for (size_t i = 0; i < num_proj; ++i) {
    scene_.projectiles.prev_position_[i] = scene_.projectiles.position_[i];
  }

  render_manager_.camera_.prev_position_ = render_manager_.camera_.position_;
}

void Game::Shutdown() {

  render_manager_.Shutdown();
}

}  // namespace rl2
