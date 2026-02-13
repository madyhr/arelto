// src/game.cpp
#include "game.h"
#include <SDL2/SDL_timer.h>
#include <SDL_keycode.h>
#include <SDL_mixer.h>
#include <SDL_mouse.h>
#include <SDL_render.h>
#include <SDL_surface.h>
#include <csignal>
#include <cstdio>
#include <iostream>
#include "audio_manager.h"
#include "constants/game.h"
#include "constants/progression_manager.h"
#include "constants/ui.h"
#include "entity.h"
#include "physics_manager.h"
#include "render_manager.h"
#include "types.h"

namespace arelto {

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

  if (!(audio_manager_.Initialize())) {
    return false;
  }

  scene_.Reset();

  if (!(Game::InitializeCamera())) {
    return false;
  }

  time_ = (float)(SDL_GetTicks64() / 1000.0f);
  SetGameState(is_running);

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

void Game::SetGameState(int game_state) {
  GameState new_state = static_cast<GameState>(game_state);

  if (new_state == in_start_screen) {
    audio_manager_.StopMusic();
  } else {
    audio_manager_.PlayMusic();
  }
  game_state_ = new_state;
}

void Game::StepGamePhysics() {
  physics_manager_.StepPhysics(scene_);
  entity_manager_.Update(scene_, physics_manager_.GetPhysicsDt());
  time_ += physics_manager_.GetPhysicsDt();
  CheckGameStateRules();
  reward_manager_.UpdateRewardTerms(scene_);
  CachePreviousState();
};

void Game::CheckGameStateRules() {
  if (entity_manager_.IsPlayerDead(scene_.player)) {
    SetGameState(is_gameover);
    return;
  }

  if (progression_manager_.CheckLevelUp(scene_.player)) {
    SetGameState(in_level_up);
    progression_manager_.GenerateLevelUpOptions(scene_);
  }
}

void Game::RenderGame(float alpha) {
  render_manager_.Render(scene_, alpha, game_status_.is_debug, time_,
                         game_state_);
};

void Game::ResetGame() {
  scene_.Reset();
  time_ = 0.0f;
  accumulator_step_ = 0.0f;
  SetGameState(is_running);
};

// Runs the game loop continuously.
void Game::RunGameLoop() {
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

        scene_.SpawnExpGem();
        RespawnEnemy(scene_.enemy, scene_.player);
        RenderGame(alpha);
        game_status_.frame_stats.print_fps_running_average(frame_time);
        break;
      }

      case in_start_screen: {
        ProcessInput();
        RenderGame(0.0f);
        break;
      }

      case is_gameover:
        break;

      case in_settings_menu: {
        float new_time = (float)(SDL_GetTicks64() / 1000.0f);
        current_time = new_time;

        if (game_status_.is_headless) {
          break;
        }
        RenderGame(0.0f);
        break;
      }

      case in_level_up: {
        float new_time = (float)(SDL_GetTicks64() / 1000.0f);
        current_time = new_time;
        ProcessLevelUpInput(SDL_GetMouseState(NULL, NULL));
        RenderGame(0.0f);
        break;
      }

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

  // The RespawnEnemy function is called outside of the accumulator loop to
  // make sure that an enemy stays dead between calls of StepGame(). This
  // could otherwise corrupt the termination signals if the enemy died,
  // respawned and died again in the same accumulator loop.
  scene_.SpawnExpGem();
  RespawnEnemy(scene_.enemy, scene_.player);
};

void Game::ProcessInput() {

  // To be able to quit while in headless mode we need to capture ctrl+C signals
  if (stop_request_) {
    SetGameState(in_shutdown);
    std::cout << "Signal received. Exiting..." << std::endl;
    return;
  };

  if (game_status_.is_headless) {
    return;
  }

  SDL_Event e;

  while (SDL_PollEvent(&e) != 0) {
    if (e.type == SDL_QUIT) {
      SetGameState(in_shutdown);
      return;
    }

    if (game_state_ == in_settings_menu) {
      ProcessSettingsMenuEvent(e);
      continue;
    }

    if (e.type == SDL_KEYDOWN) {
      switch (e.key.keysym.sym) {
        case SDLK_q:
          SetGameState(in_shutdown);
          std::cout << "Key 'q' pressed! Exiting..." << std::endl;
          break;

        case SDLK_r:
          if (game_state_ == is_gameover) {
            ResetGame();
          }
          break;

        case SDLK_ESCAPE:
          if (game_state_ == is_running) {
            SetGameState(in_settings_menu);
          } else if (game_state_ == in_settings_menu) {
            SetGameState(is_running);
          } else {
            ResetGame();
            SetGameState(in_start_screen);
          }
          break;

        case SDLK_m:
          audio_manager_.ToggleMusic();
          break;

        case SDLK_COMMA:
          audio_manager_.DecreaseMusicVolume();
          break;

        case SDLK_PERIOD:
          audio_manager_.IncreaseMusicVolume();
          break;
      }

    } else if (e.type == SDL_MOUSEBUTTONDOWN) {
      if (game_state_ == in_start_screen) {
        int mouse_x = e.button.x;
        int mouse_y = e.button.y;

        int btn_w = kBeginButtonWidth;
        int btn_h = kBeginButtonHeight;
        int btn_x = kBeginButtonX;
        int btn_y = kBeginButtonY;

        if (mouse_x >= btn_x && mouse_x <= btn_x + btn_w && mouse_y >= btn_y &&
            mouse_y <= btn_y + btn_h) {
          SetGameState(is_running);
          std::cout << "Game Started!" << std::endl;
        }
      }
    }
  }

  int cursor_pos_x, cursor_pos_y;
  uint32_t mouse_state = SDL_GetMouseState(&cursor_pos_x, &cursor_pos_y);

  if (game_state_ == in_level_up) {
    ProcessLevelUpInput(SDL_GetMouseState(NULL, NULL));
    return;
  }

  if (game_state_ == in_settings_menu) {
    ProcessSettingsMenuInput(mouse_state);
    return;
  }

  cursor_position_ = {
      (float)(cursor_pos_x + render_manager_.camera_.position_.x),
      (float)(cursor_pos_y + render_manager_.camera_.position_.y)};

  ProcessPlayerInput(mouse_state);
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

void Game::ProcessSettingsMenuEvent(const SDL_Event& e) {
  if (e.type == SDL_KEYDOWN) {
    switch (e.key.keysym.sym) {
      case SDLK_q:
        SetGameState(in_shutdown);
        std::cout << "Key 'q' pressed! Exiting..." << std::endl;
        break;
      case SDLK_ESCAPE:
        SetGameState(is_running);
        break;
      case SDLK_m:
        audio_manager_.ToggleMusic();
        break;
      case SDLK_COMMA:
        audio_manager_.DecreaseMusicVolume();
        break;
      case SDLK_PERIOD:
        audio_manager_.IncreaseMusicVolume();
        break;
    }
  } else if (e.type == SDL_MOUSEBUTTONDOWN) {
    int mouse_x = e.button.x;
    int mouse_y = e.button.y;
    int btn_w = kSettingsMenuButtonWidth;
    int btn_h = kSettingsMenuButtonHeight;

    int mute_x = kSettingsMenuX + kSettingsMenuButtonX;
    int mute_y = kSettingsMenuY + kSettingsMenuMuteY;
    if (mouse_x >= mute_x && mouse_x <= mute_x + btn_w && mouse_y >= mute_y &&
        mouse_y <= mute_y + btn_h) {
      audio_manager_.ToggleMusic();
    }

    int main_x = kSettingsMenuX + kSettingsMenuMainMenuX;
    int main_y = kSettingsMenuY + kSettingsMenuMainMenuY;
    if (mouse_x >= main_x && mouse_x <= main_x + btn_w && mouse_y >= main_y &&
        mouse_y <= main_y + btn_h) {
      ResetGame();
      SetGameState(in_start_screen);
    }

    int resume_x = kSettingsMenuX + kSettingsMenuResumeX;
    int resume_y = kSettingsMenuY + kSettingsMenuResumeY;
    if (mouse_x >= resume_x && mouse_x <= resume_x + btn_w &&
        mouse_y >= resume_y && mouse_y <= resume_y + btn_h) {
      SetGameState(is_running);
    }
  }
}

void Game::ProcessSettingsMenuInput(uint32_t mouse_state) {
  if (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
    int mouse_x, mouse_y;
    SDL_GetMouseState(&mouse_x, &mouse_y);

    int slider_x = kSettingsMenuX + kSettingsMenuVolumeSliderX;
    int slider_y = kSettingsMenuY + kSettingsMenuVolumeSliderY;
    int slider_w = kSettingsMenuVolumeSliderWidth;
    int slider_h = kSettingsMenuVolumeSliderHeight;

    if (mouse_x >= slider_x && mouse_x <= slider_x + slider_w &&
        mouse_y >= slider_y - 10 && mouse_y <= slider_y + slider_h + 10) {
      float percent = static_cast<float>(mouse_x - slider_x) / slider_w;
      audio_manager_.SetMusicVolume(percent);
    }
  }

  // The SDL_mixer music volume goes from 0 to 128.
  render_manager_.UpdateSettingsMenuState(
      audio_manager_.GetMusicVolume() * 128.0f, audio_manager_.IsMusicMuted());
}

// This function processes inputs during a level up state and applies the
// upgrade option according to which button was clicked.
void Game::ProcessLevelUpInput(uint32_t mouse_state) {
  if (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
    int total_width = kNumUpgradeOptions * kLevelUpCardWidth +
                      (kNumUpgradeOptions - 1) * kLevelUpCardGap;
    int start_x = (kWindowWidth - total_width) / 2;
    int start_y = (kWindowHeight - kLevelUpCardHeight) / 2;

    int mouse_x, mouse_y;
    SDL_GetMouseState(&mouse_x, &mouse_y);

    int selected_card_idx = -1;

    int button_top = start_y + kLevelUpButtonOffsetY;
    int button_bottom = button_top + kLevelUpButtonHeight;

    if (mouse_y >= button_top && mouse_y <= button_bottom) {
      for (int i = 0; i < kNumUpgradeOptions; ++i) {
        int card_x = start_x + i * (kLevelUpCardWidth + kLevelUpCardGap);
        int button_x = card_x + (kLevelUpCardWidth - kLevelUpButtonWidth) / 2;
        int button_right = button_x + kLevelUpButtonWidth;

        if (mouse_x >= button_x && mouse_x <= button_right) {
          selected_card_idx = i;
          break;
        }
      }
    }

    if (selected_card_idx != -1) {
      progression_manager_.ApplyUpgrade(scene_, selected_card_idx);
      SetGameState(is_running);
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
  audio_manager_.Shutdown();
}

}  // namespace arelto
