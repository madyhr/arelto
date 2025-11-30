// src/game.cpp
#include "game.h"
#include <SDL2/SDL_timer.h>
#include <SDL_render.h>
#include <SDL_surface.h>
#include <algorithm>
#include <array>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <vector>
#include "constants.h"
#include "entity.h"
#include "random.h"
#include "types.h"

namespace {
volatile std::sig_atomic_t g_stop_request = 0;
}

void SignalHandler(int signal) {
  g_stop_request = 1;
}

namespace rl2 {

Game::Game(){};

Game::~Game() {
  Game::Shutdown();
}

int Game::GetGameState() {
  return static_cast<int>(game_state_);
};

bool Game::Initialize() {

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGKILL, SignalHandler);
  game_status_.in_debug_mode = true;
  game_status_.in_headless_mode = false;

  if (!(Game::InitializeResources())) {
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

bool Game::InitializeResources() {

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL could not initialize! SDL Error: " << SDL_GetError()
              << std::endl;
    return false;
  }

  if (game_status_.in_headless_mode) {
    return true;
  }

  resources_.window =
      SDL_CreateWindow("RL2", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                       kWindowWidth, kWindowHeight, SDL_WINDOW_SHOWN);

  if (resources_.window == nullptr) {
    std::cerr << "Window could not be created: " << SDL_GetError() << std::endl;
    return false;
  }

  resources_.renderer =
      SDL_CreateRenderer(resources_.window, -1, SDL_RENDERER_ACCELERATED);

  if (resources_.renderer == nullptr) {
    std::cerr << "Renderer could not be created: " << SDL_GetError()
              << std::endl;
    return false;
  }

  int img_flags = IMG_INIT_PNG;
  if (!(IMG_Init(img_flags) & img_flags)) {
    std::cerr << "SDL Images could not be initialized: " << SDL_GetError()
              << std::endl;
    return false;
  }

  resources_.tile_manager.SetupTileMap();
  resources_.tile_manager.SetupTiles();
  resources_.tile_manager.SetupTileSelector();

  resources_.tile_texture = resources_.tile_manager.GetTileTexture(
      "assets/dungeon_floor_tiles_tall.bmp", resources_.renderer);
  resources_.player_texture = IMG_LoadTexture(
      resources_.renderer, "assets/textures/wizard_sprite_sheet_with_idle.png");
  // resources_.enemy_texture = IMG_LoadTexture(
  // resources_.renderer, "assets/textures/goblin_sprite_sheet.png");
  resources_.enemy_texture = IMG_LoadTexture(
      resources_.renderer, "assets/textures/tentacle_being_sprite_sheet.png");
  resources_.projectile_textures.push_back(IMG_LoadTexture(
      resources_.renderer, "assets/textures/fireball_sprite_sheet.png"));
  resources_.projectile_textures.push_back(IMG_LoadTexture(
      resources_.renderer, "assets/textures/frostbolt_sprite_sheet.png"));

  if (resources_.tile_texture == nullptr ||
      resources_.player_texture == nullptr ||
      resources_.enemy_texture == nullptr ||
      std::any_of(
          resources_.projectile_textures.begin(),
          resources_.projectile_textures.end(),
          [](SDL_Texture* sdl_texture) { return sdl_texture == nullptr; })) {
    std::cerr << "One or more textures could not be loaded: " << SDL_GetError()
              << std::endl;
    return false;
  }

  resources_.map_layout = {0, 0, kMapWidth, kMapHeight};
  return true;
};

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
  camera_.position_.x = player_centroid.x - 0.5f * kWindowWidth;
  camera_.position_.y = player_centroid.y - 0.5f * kWindowHeight;

  return true;
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
//
// void Game::Render(float alpha) {
//   if (game_status_.in_headless_mode) {
//     return;
//   }
//   // Setting alpha to 1.0f to always render the latest state.
//   Game::GenerateOutput(alpha);
// }
//
void Game::RunGameLoop() {
  // Runs the game loop continuously. Acts as a way to only run the game.

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
  if (g_stop_request) {
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

int Game::GetObservationSize() {
  return 2 +                  // player position
         (kNumEnemies * 2) +  // enemy position: x,y
         (kNumEnemies * 2) +  // enemy velocity: x,y
         (kNumEnemies * 2) +  // enemy size: w,h
         (kNumEnemies) +      // enemy health_points
         (kNumEnemies) +      // enemy inv mass
         (kNumEnemies) +      // enemy movement speed
         (kNumEnemies *
          enemy_.occupancy_map[0].kTotalCells);  // enemy occupancy map
};

// void Game::FillObservationBuffer(py::array_t<float> buffer) {
void Game::FillObservationBuffer(float* buffer_ptr, int buffer_size) {

  if (buffer_size != GetObservationSize()) {
    throw std::runtime_error("Buffer size mismatch");
  };

  int idx = 0;

  buffer_ptr[idx++] = player_.position_.x;
  buffer_ptr[idx++] = player_.position_.y;

  for (const Vector2D& enemy_pos : enemy_.position) {
    buffer_ptr[idx++] = enemy_pos.x;
    buffer_ptr[idx++] = enemy_pos.y;
  }

  for (const Vector2D& enemy_pos : enemy_.position) {
    buffer_ptr[idx++] = enemy_pos.x;
    buffer_ptr[idx++] = enemy_pos.y;
  }

  for (const Size& enemy_size : enemy_.size) {
    buffer_ptr[idx++] = static_cast<float>(enemy_size.width);
    buffer_ptr[idx++] = static_cast<float>(enemy_size.height);
  }

  for (const int& enemy_health : enemy_.health_points) {
    buffer_ptr[idx++] = static_cast<float>(enemy_health);
  }

  for (const float& enemy_inv_mass : enemy_.inv_mass) {
    buffer_ptr[idx++] = enemy_inv_mass;
  }

  for (const float& enemy_movement_speed : enemy_.movement_speed) {
    buffer_ptr[idx++] = enemy_movement_speed;
  }

  for (int i = 0; i < kNumEnemies; ++i) {
    const EntityType* map_data = enemy_.occupancy_map[i].Data();
    size_t total_cells = enemy_.occupancy_map[i].kTotalCells;

    for (size_t k = 0; k < total_cells; ++k) {
      buffer_ptr[idx++] = static_cast<float>(map_data[k]);
    }
  }

  return;
};

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
