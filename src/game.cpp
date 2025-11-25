// src/game.cpp
#include "game.h"
#include <SDL2/SDL_timer.h>
#include <SDL_render.h>
#include <SDL_surface.h>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include "collision.h"
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
  resources_.player_texture = IMG_LoadTexture(
      resources_.renderer, "assets/textures/wizard_sprite_sheet_with_idle.png");
  resources_.enemy_texture = IMG_LoadTexture(
      resources_.renderer, "assets/textures/goblin_sprite_sheet.png");
  resources_.projectile_textures.push_back(
      IMG_LoadTexture(resources_.renderer, "assets/textures/fireball.png"));
  resources_.projectile_textures.push_back(
      IMG_LoadTexture(resources_.renderer, "assets/textures/frostbolt.png"));

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

void Game::RunGameLoop() {
  while (is_running_) {
    float current_time = (float)(SDL_GetTicks64() / 1000.0f);
    dt = current_time - time_;
    Game::Update();
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

void Game::Update() {

  Game::ProcessInput();
  Game::UpdatePlayerPosition(dt);
  Game::UpdateEnemyPosition(dt);
  Game::UpdateProjectilePosition(dt);
  Game::HandleCollisions();
  Game::HandleOutOfBounds();

  Game::UpdateCameraPosition();
  UpdateEnemyStatus(enemy_, player_);
  game_status_.frame_stats.print_fps_running_average(dt);

  Game::GenerateOutput();
  projectiles_.DestroyProjectiles();
}

void Game::UpdatePlayerPosition(float dt) {
  Vector2D delta_pos =
      player_.velocity_.Normalized() * (player_.stats_.movement_speed * dt);
  player_.position_ += delta_pos;
};

void Game::UpdateEnemyPosition(float dt) {
  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemy_.is_alive[i]) {
      Vector2D distance_vector = player_.position_ - enemy_.position[i];
      enemy_.velocity[i] = distance_vector.Normalized();
      enemy_.position[i] += enemy_.velocity[i] * enemy_.movement_speed[i] * dt;
    }
  }
};

void Game::UpdateProjectilePosition(float dt) {
  size_t num_projectiles = projectiles_.GetNumProjectiles();
  if (num_projectiles == 0) {
    return;
  };

  for (int i = 0; i < num_projectiles; ++i) {
    projectiles_.position_[i] +=
        projectiles_.direction_[i] * projectiles_.speed_[i] * dt;
  };
};

void Game::HandleCollisions() {
  HandleCollisionsSAP(player_, enemy_, projectiles_);
};

void Game::HandleOutOfBounds() {
  rl2::HandlePlayerOOB(player_);
  rl2::HandleEnemyOOB(enemy_);
  rl2::HandleProjectileOOB(projectiles_);
};

void Game::UpdateCameraPosition() {
  Vector2D player_centroid =
      GetCentroid(player_.position_, player_.stats_.size);
  camera_.position_.x = player_centroid.x - 0.5f * kWindowWidth;
  camera_.position_.y = player_centroid.y - 0.5f * kWindowHeight;
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
  RenderPlayer();

  int num_enemy_vertices = SetupEnemyGeometry();
  RenderEnemies(num_enemy_vertices);

  SetupProjectileGeometry();
  RenderProjectiles();

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

void Game::RenderPlayer() {
  SDL_Rect player_render_box = {
      (int)(player_.position_.x - camera_.position_.x),
      (int)(player_.position_.y - camera_.position_.y),
      (int)player_.stats_.size.width, (int)player_.stats_.size.height};

  bool is_standing_still = player_.velocity_.Norm() < 1e-3;
  bool is_facing_right = player_.last_horizontal_velocity_ >= 0.0f;

  SDL_Rect src_rect;
  src_rect.w = kPlayerSpriteCellWidth;
  src_rect.h = kPlayerSpriteCellHeight;
  src_rect.y = is_standing_still ? 0 : kPlayerSpriteCellHeight;
  src_rect.x = ((SDL_GetTicks64() / 150) % kPlayerNumSpriteCells) * src_rect.w;


  SDL_RendererFlip is_flipped =
      is_facing_right ? SDL_FLIP_NONE : SDL_FLIP_HORIZONTAL;

  SDL_RenderCopyEx(resources_.renderer, resources_.player_texture, &src_rect,
                   &player_render_box, 0.0, nullptr, is_flipped);

  if (player_.velocity_.x != 0) {
    player_.last_horizontal_velocity_ = player_.velocity_.x;
  }
};

int Game::SetupEnemyGeometry() {
  // The return type is int as we need to know how many vertices to actually
  // render when we call SDLRenderGeometry. So we traverse the enemies struct
  // and keep count of the total number of active vertices.

  int current_vertex_idx = 0;

  float cell_uv_width = 1.0f / (float)kEnemyNumSpriteCells;

  for (int i = 0; i < kNumEnemies; ++i) {
    if (!enemy_.is_alive[i]) {
      continue;
    };

    float x = enemy_.position[i].x - camera_.position_.x;
    float y = enemy_.position[i].y - camera_.position_.y;
    float w = enemy_.size[i].width;
    float h = enemy_.size[i].height;

    int frame_idx = (SDL_GetTicks64() / kEnemyAnimationFrameDuration) %
                    kEnemyNumSpriteCells;

    float u_left = frame_idx * cell_uv_width;
    float u_right = u_left + cell_uv_width;
    float v_top = kTexCoordTop;
    float v_bottom = kTexCoordBottom;

    bool is_facing_right = enemy_.last_horizontal_velocity[i] > 0;

    float vertex_left = is_facing_right ? u_left : u_right;
    float vertex_right = is_facing_right ? u_right : u_left;

    // --- Vertices for Triangle 1 (Top-Left, Bottom-Left, Bottom-Right) ---
    // 1. Top-Left
    enemies_vertices_[current_vertex_idx + 0] = {
        {x, y}, {255, 255, 255, 255}, {vertex_left, v_top}};
    // 2. Bottom-Left
    enemies_vertices_[current_vertex_idx + 1] = {
        {x, y + h}, {255, 255, 255, 255}, {vertex_left, v_bottom}};
    // 3. Bottom-Right
    enemies_vertices_[current_vertex_idx + 2] = {
        {x + w, y + h}, {255, 255, 255, 255}, {vertex_right, v_bottom}};
    // --- Vertices for Triangle 2 (Top-Left, Bottom-Right, Top-Right) ---
    // 4. Top-Left (Repeat)
    enemies_vertices_[current_vertex_idx + 3] =
        enemies_vertices_[current_vertex_idx + 0];  // Same as vertex 1
    // 5. Bottom-Right (Repeat)
    enemies_vertices_[current_vertex_idx + 4] =
        enemies_vertices_[current_vertex_idx + 2];  // Same as vertex 3
    // 6. Top-Right
    enemies_vertices_[current_vertex_idx + 5] = {
        {x + w, y}, {255, 255, 255, 255}, {vertex_right, v_top}};

    current_vertex_idx += kEnemyVertices;

    if (enemy_.velocity[i].x != 0) {
      enemy_.last_horizontal_velocity[i] = enemy_.velocity[i].x;
    }
  }
  return current_vertex_idx;
};

void Game::RenderEnemies(int num_vertices) {
  // We use the number of vertices calculated during the setup of the enemy
  // geometry to render the vertices.
  SDL_RenderGeometry(resources_.renderer, resources_.enemy_texture,
                     enemies_vertices_, num_vertices, nullptr, 0);
};

void Game::SetupProjectileGeometry() {
  projectile_vertices_grouped_.clear();
  size_t num_projectiles = projectiles_.GetNumProjectiles();
  if (num_projectiles == 0) {
    return;
  }

  for (int i = 0; i < num_projectiles; ++i) {
    float x = projectiles_.position_[i].x - camera_.position_.x;
    float y = projectiles_.position_[i].y - camera_.position_.y;
    float w = projectiles_.size_[i].width;
    float h = projectiles_.size_[i].height;
    int texture_id = projectiles_.proj_id_[i];

    SDL_Vertex vertices[kProjectileVertices];

    // --- Vertices for Triangle 1 (Top-Left, Bottom-Left, Bottom-Right) ---
    // 1. Top-Left
    vertices[0] = {{x, y}, {255, 255, 255, 255}, {kTexCoordLeft, kTexCoordTop}};
    // 2. Bottom-Left
    vertices[1] = {
        {x, y + h}, {255, 255, 255, 255}, {kTexCoordLeft, kTexCoordBottom}};
    // 3. Bottom-Right
    vertices[2] = {{x + w, y + h},
                   {255, 255, 255, 255},
                   {kTexCoordRight, kTexCoordBottom}};
    // --- Vertices for Triangle 2 (Top-Left, Bottom-Right, Top-Right) ---
    // 4. Top-Left (Repeat)
    vertices[3] = vertices[0];  // Same as vertex 1
                                // 5. Bottom-Right (Repeat)
    vertices[4] = vertices[2];  // Same as vertex 3
    // 6. Top-Right
    vertices[5] = {
        {x + w, y}, {255, 255, 255, 255}, {kTexCoordRight, kTexCoordTop}};

    for (int j = 0; j < kProjectileVertices; ++j) {
      projectile_vertices_grouped_[texture_id].push_back(vertices[j]);
    }
  }
};

void Game::RenderProjectiles() {

  for (const auto& pair : projectile_vertices_grouped_) {
    int texture_id = pair.first;
    const std::vector<SDL_Vertex>& vertices = pair.second;
    if (texture_id >= 0 && texture_id < resources_.projectile_textures.size()) {
      SDL_RenderGeometry(resources_.renderer,
                         resources_.projectile_textures[texture_id],
                         vertices.data(), (int)vertices.size(), nullptr, 0);
    };
  };
};

void Game::GetModelObservation() {
  std::vector<float> obs_buffer; 
   

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
