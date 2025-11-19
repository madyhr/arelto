// src/game.cpp
#include "game.h"
#include <SDL2/SDL_timer.h>
#include <SDL_render.h>
#include <SDL_surface.h>
#include <cstdio>
#include <iostream>
#include "constants.h"
#include "entity.h"
#include "game_math.h"
#include "random.h"
#include "types.h"

namespace rl2 {

Game::Game(){};

Game::~Game() {
  Game::Shutdown();
}

bool Game::Initialize() {
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
  is_running_ = true;

  return true;
}

bool Game::InitializeResources() {

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL could not initialize! SDL Error: " << SDL_GetError()
              << std::endl;
    return false;
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
    "assets/grassy_tiles.bmp", resources_.renderer);

  // resources_.map_texture =
  //     IMG_LoadTexture(resources_.renderer, "assets/textures/grassy_plains.png");
  resources_.player_texture =
    IMG_LoadTexture(resources_.renderer, "assets/textures/wizard.png");
  resources_.enemy_texture =
    IMG_LoadTexture(resources_.renderer, "assets/textures/goblin.png");
  resources_.projectile_texture =
    IMG_LoadTexture(resources_.renderer, "assets/textures/fireball.png");

  if (
    // resources_.map_texture == nullptr ||
    resources_.player_texture == nullptr ||
    resources_.enemy_texture == nullptr ||
    resources_.projectile_texture == nullptr) {
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
  return true;
};

bool Game::InitializeEnemies() {
  std::fill(enemy_.are_alive.begin(), enemy_.are_alive.end(), true);
  std::fill(enemy_.movement_speeds.begin(), enemy_.movement_speeds.end(),
            kEnemySpeed);
  std::fill(enemy_.sizes.begin(), enemy_.sizes.end(),
            Size{kEnemyHeight, kEnemyWidth});
  std::fill(enemy_.health_points.begin(), enemy_.health_points.end(),
            kEnemyHealth);
  std::fill(enemy_.inv_masses.begin(), enemy_.inv_masses.end(), kEnemyInvMass);

  int max_x = kMapWidth - kEnemyWidth;
  int max_y = kMapHeight - kEnemyHeight;

  for (int i = 0; i < kNumEnemies; ++i) {
    Vector2D potential_pos;

    do {
      potential_pos = {(float)GenerateRandomInt(0, max_x),
                       (float)GenerateRandomInt(0, max_y)};

    } while (CalculateVector2dDistance(potential_pos, player_.position_) <
             kEnemyMinimumInitialDistance);

    enemy_.positions[i] = potential_pos;
    enemy_.movement_speeds[i] += GenerateRandomInt(1, 100);
    enemy_.sizes[i].height += GenerateRandomInt(1, 50);
    enemy_.sizes[i].width += GenerateRandomInt(1, 50);
  };

  SDL_Color red = {255, 0, 0, 255};
  for (int i = 0; i < kTotalEnemyVertices; ++i) {
    enemies_vertices_[i].color = red;
  }
  return true;
};

bool Game::InitializeCamera() {
  camera_.position_.x = player_.position_.x - 0.5f * kWindowWidth;
  camera_.position_.y = player_.position_.y - 0.5f * kWindowHeight;

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

void Game::RunGameLoop() {
  while (is_running_) {
    float current_time = (float)(SDL_GetTicks64() / 1000.0f);
    dt = current_time - time_;

    Game::ProcessInput();
    Game::Update();
    Game::GenerateOutput();

    time_ = current_time;
  }
};

void Game::ProcessInput() {

  SDL_Event e;

  while (SDL_PollEvent(&e) != 0) {
    if (e.type == SDL_QUIT) {
      is_running_ = false;
    } else if (e.type == SDL_KEYDOWN) {

      switch (e.key.keysym.sym) {
        case SDLK_q:
          is_running_ = false;
          std::cout << "Key 'q' pressed! Exiting..." << std::endl;
          break;
      }
    }
  }
  Game::ProcessPlayerInput();
}

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
    bool fireball_is_ready = time_ >= player_.fireball_.GetReadyTime();
    if (fireball_is_ready) {
      Vector2D fireball_velocity = NormalizeVector2D(player_.velocity_);
      if (player_.velocity_.Norm() < 1e-3) {
        fireball_velocity.x = GenerateRandomFloat(-1.0f, 1.0f) * kFireballSpeed;
        fireball_velocity.y = GenerateRandomFloat(-1.0f, 1.0f) * kFireballSpeed;
        fireball_velocity = NormalizeVector2D(fireball_velocity);
      }
      ProjectileData fireball = {0,
                                 player_.position_,
                                 fireball_velocity,
                                 kFireballSpeed,
                                 {kFireballWidth, kFireballHeight},
                                 0.0f};
      projectiles_.AddProjectile(fireball);
      player_.fireball_.time_of_last_use = time_;
    }
  }
}

void Game::Update() {

  Game::UpdatePlayerPosition(dt);
  Game::UpdateEnemyPosition(dt);
  Game::UpdateProjectilePosition(dt);
  Game::HandleCollisions();
  Game::HandleOutOfBounds();

  Game::UpdateCameraPosition();

  game_status_.frame_stats.print_fps_running_average(dt);
}

void Game::UpdatePlayerPosition(float dt) {

  float player_velocity_magnitude = player_.velocity_.Norm();
  if (player_velocity_magnitude > 1.0f) {
    player_.velocity_.x /= player_velocity_magnitude;
    player_.velocity_.y /= player_velocity_magnitude;
  }

  player_.position_.x +=
    player_.velocity_.x * player_.stats_.movement_speed * dt;
  player_.position_.y +=
    player_.velocity_.y * player_.stats_.movement_speed * dt;
};

void Game::UpdateEnemyPosition(float dt) {
  for (int i = 0; i < kNumEnemies; ++i) {
    float dx = player_.position_.x - enemy_.positions[i].x;
    float dy = player_.position_.y - enemy_.positions[i].y;
    float distance_to_player = std::hypot(dx, dy);
    enemy_.velocities[i].x = dx / (distance_to_player + 1e-6);
    enemy_.velocities[i].y = dy / (distance_to_player + 1e-6);
    enemy_.positions[i].x +=
      enemy_.velocities[i].x * enemy_.movement_speeds[i] * dt;
    enemy_.positions[i].y +=
      enemy_.velocities[i].y * enemy_.movement_speeds[i] * dt;
  }
};

void Game::UpdateProjectilePosition(float dt) {
  size_t num_projectiles = projectiles_.GetNumProjectiles();
  if (num_projectiles == 0) {
    return;
  };

  for (int i = 0; i < num_projectiles; ++i) {
    projectiles_.positions_[i].x +=
      projectiles_.velocities_[i].x * projectiles_.speeds_[i] * dt;
    projectiles_.positions_[i].y +=
      projectiles_.velocities_[i].y * projectiles_.speeds_[i] * dt;
  };
};

void Game::HandleCollisions() {
  rl2::HandleCollisionsSAP(player_, enemy_);
};

void Game::HandleOutOfBounds() {
  rl2::HandlePlayerOOB(player_);
  rl2::HandleEnemyOOB(enemy_);
  rl2::HandleProjectileOOB(projectiles_);
};

void Game::UpdateCameraPosition() {
  camera_.position_.x = player_.position_.x - 0.5f * kWindowWidth;
  camera_.position_.y = player_.position_.y - 0.5f * kWindowHeight;
  if (camera_.position_.x < 0) {
    camera_.position_.x = 0.0f;
  };
  if (camera_.position_.y < 0) {
    camera_.position_.y = 0.0f;
  };
  if (camera_.position_.x > (kMapWidth - kWindowWidth)) {
    camera_.position_.x = kMapWidth - kWindowWidth;
  }
  if (camera_.position_.y > (kMapHeight - kWindowHeight)) {
    camera_.position_.y = kMapHeight - kWindowHeight;
  }
};

void Game::GenerateOutput() {
  SDL_SetRenderDrawColor(resources_.renderer, 0x00, 0x00, 0x00, 0xFF);
  SDL_RenderClear(resources_.renderer);
  RenderTiledMap();
  // SDL_RenderCopy(resources_.renderer, resources_.map_texture, NULL,
  //                &camera_render_box);
  SDL_Rect player_render_box = {
    (int)(player_.position_.x - camera_.position_.x),
    (int)(player_.position_.y - camera_.position_.y),
    (int)player_.stats_.size.width, (int)player_.stats_.size.height};

  SDL_RenderCopy(resources_.renderer, resources_.player_texture, NULL,
                 &player_render_box);
  SetupEnemyGeometry();
  SDL_RenderGeometry(resources_.renderer, resources_.enemy_texture,
                     enemies_vertices_, kTotalEnemyVertices, nullptr, 0);

  SetupProjectileGeometry();

  size_t num_projectiles = projectiles_.GetNumProjectiles();
  int num_proj_vertices = num_projectiles * kProjectileVertices;
  SDL_RenderGeometry(resources_.renderer, resources_.projectile_texture,
                     projectiles_vertices_.data(), num_proj_vertices, nullptr,
                     0);
  // For debugging render boxes
  // SDL_SetRenderDrawColor(resources_.renderer, 0, 0, 0, 255);
  // SDL_RenderFillRect(resources_.renderer, &player_render_box);
  // SDL_RenderGeometry(resources_.renderer, nullptr, enemy_vertices_,
  //                    kTotalEnemyVertices, nullptr, 0);
  //
  //
  SDL_RenderPresent(resources_.renderer);
};

void Game::RenderTiledMap() {
  int top_left_tile_x = (int)(camera_.position_.x / kTileSize);
  int top_left_tile_y = (int)(camera_.position_.y / kTileSize);
  int bottom_right_tile_x =
    (int)std::ceil((camera_.position_.x + kWindowWidth) / kTileSize);
  int bottom_right_tile_y =
    (int)std::ceil((camera_.position_.y + kWindowHeight) / kTileSize);
  int start_x = std::max(0, top_left_tile_x);
  int end_x = std::min(kNumTilesX, bottom_right_tile_x);

  int start_y = std::max(0, top_left_tile_y);
  int end_y = std::min(kNumTilesY, bottom_right_tile_y);

  for (int i = start_x; i < end_x; ++i) {
    for (int j = start_y; j < end_y; ++j) {
      SDL_Rect render_rect = resources_.tile_manager.tiles_[i][j];
      render_rect.x -= (int)camera_.position_.x;
      render_rect.y -= (int)camera_.position_.y;
      int tile_id = resources_.tile_manager.tile_map_[i][j];
      const SDL_Rect& source_rect =
        resources_.tile_manager.select_tiles_[tile_id];
      SDL_RenderCopy(resources_.renderer, resources_.tile_texture, &source_rect,
                     &render_rect);
    }
  }
};

void Game::SetupEnemyGeometry() {
  for (int i = 0; i < kNumEnemies; ++i) {
    float x = enemy_.positions[i].x - camera_.position_.x;
    float y = enemy_.positions[i].y - camera_.position_.y;
    float w = enemy_.sizes[i].width;
    float h = enemy_.sizes[i].height;

    int vertex_offset = i * kEnemyVertices;

    // --- Vertices for Triangle 1 (Top-Left, Bottom-Left, Bottom-Right) ---
    // 1. Top-Left
    enemies_vertices_[vertex_offset + 0] = {
      {x, y}, {255, 255, 255, 255}, {kTexCoordLeft, kTexCoordTop}};
    // 2. Bottom-Left
    enemies_vertices_[vertex_offset + 1] = {
      {x, y + h}, {255, 255, 255, 255}, {kTexCoordLeft, kTexCoordBottom}};
    // 3. Bottom-Right
    enemies_vertices_[vertex_offset + 2] = {
      {x + w, y + h}, {255, 255, 255, 255}, {kTexCoordRight, kTexCoordBottom}};
    // --- Vertices for Triangle 2 (Top-Left, Bottom-Right, Top-Right) ---
    // 4. Top-Left (Repeat)
    enemies_vertices_[vertex_offset + 3] =
      enemies_vertices_[vertex_offset + 0];  // Same as vertex 1
    // 5. Bottom-Right (Repeat)
    enemies_vertices_[vertex_offset + 4] =
      enemies_vertices_[vertex_offset + 2];  // Same as vertex 3
    // 6. Top-Right
    enemies_vertices_[vertex_offset + 5] = {
      {x + w, y}, {255, 255, 255, 255}, {kTexCoordRight, kTexCoordTop}};
  }
};

void Game::SetupProjectileGeometry() {
  size_t num_projectiles = projectiles_.GetNumProjectiles();
  if (num_projectiles == 0) {
    return;
  }

  size_t required_vertices = num_projectiles * kProjectileVertices;
  projectiles_vertices_.resize(required_vertices);

  for (int i = 0; i < num_projectiles; ++i) {
    float x = projectiles_.positions_[i].x - camera_.position_.x;
    float y = projectiles_.positions_[i].y - camera_.position_.y;
    float w = projectiles_.sizes_[i].width;
    float h = projectiles_.sizes_[i].height;

    int vertex_offset = i * kProjectileVertices;

    // --- Vertices for Triangle 1 (Top-Left, Bottom-Left, Bottom-Right) ---
    // 1. Top-Left
    projectiles_vertices_[vertex_offset + 0] = {
      {x, y}, {255, 255, 255, 255}, {kTexCoordLeft, kTexCoordTop}};
    // 2. Bottom-Left
    projectiles_vertices_[vertex_offset + 1] = {
      {x, y + h}, {255, 255, 255, 255}, {kTexCoordLeft, kTexCoordBottom}};
    // 3. Bottom-Right
    projectiles_vertices_[vertex_offset + 2] = {
      {x + w, y + h}, {255, 255, 255, 255}, {kTexCoordRight, kTexCoordBottom}};
    // --- Vertices for Triangle 2 (Top-Left, Bottom-Right, Top-Right) ---
    // 4. Top-Left (Repeat)
    projectiles_vertices_[vertex_offset + 3] =
      projectiles_vertices_[vertex_offset + 0];  // Same as vertex 1
                                                 // 5. Bottom-Right (Repeat)
    projectiles_vertices_[vertex_offset + 4] =
      projectiles_vertices_[vertex_offset + 2];  // Same as vertex 3
    // 6. Top-Right
    projectiles_vertices_[vertex_offset + 5] = {
      {x + w, y}, {255, 255, 255, 255}, {kTexCoordRight, kTexCoordTop}};
  }
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
