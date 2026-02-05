// src/render_manager.cpp
#include "render_manager.h"
#include <SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_timer.h>
#include <SDL_rect.h>
#include <SDL_render.h>
#include <algorithm>
#include <iostream>
#include "constants/enemy.h"
#include "constants/exp_gem.h"
#include "constants/game.h"
#include "constants/map.h"
#include "constants/player.h"
#include "constants/progression_manager.h"
#include "constants/projectile.h"
#include "constants/ray_caster.h"
#include "constants/render.h"
#include "constants/ui.h"
#include "entity.h"
#include "scene.h"
#include "types.h"
#include "ui_manager.h"

namespace arelto {

RenderManager::RenderManager() {};
RenderManager::~RenderManager() {
  Shutdown();
};

bool RenderManager::Initialize(bool is_headless) {

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL could not initialize! SDL Error: " << SDL_GetError()
              << std::endl;
    return false;
  }

  if (is_headless) {
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

  if (TTF_Init() == -1) {
    std::cerr << "SDL_ttf could not be initialized: " << TTF_GetError()
              << std::endl;
    return false;
  }

  resources_.tile_manager.SetupTileMap();
  resources_.tile_manager.SetupTiles();
  resources_.tile_manager.SetupTileSelector();

  ui_manager_.SetupUI();

  resources_.tile_texture = resources_.tile_manager.GetTileTexture(
      "assets/dungeon_floor_tiles_tall.bmp", resources_.renderer);
  resources_.player_texture = IMG_LoadTexture(
      resources_.renderer, "assets/textures/wizard_sprite_sheet_with_idle.png");
  resources_.enemy_texture = IMG_LoadTexture(
      resources_.renderer, "assets/textures/tentacle_being_sprite_sheet.png");
  resources_.projectile_textures.push_back(IMG_LoadTexture(
      resources_.renderer, "assets/textures/fireball_sprite_sheet.png"));
  resources_.projectile_textures.push_back(IMG_LoadTexture(
      resources_.renderer, "assets/textures/frostbolt_sprite_sheet.png"));
  resources_.gem_textures.push_back(IMG_LoadTexture(
      resources_.renderer, "assets/textures/exp_gem_common.png"));
  resources_.gem_textures.push_back(
      IMG_LoadTexture(resources_.renderer, "assets/textures/exp_gem_rare.png"));
  resources_.gem_textures.push_back(
      IMG_LoadTexture(resources_.renderer, "assets/textures/exp_gem_epic.png"));
  resources_.gem_textures.push_back(IMG_LoadTexture(
      resources_.renderer, "assets/textures/exp_gem_legendary.png"));
  resources_.ui_resources.level_indicator_texture = IMG_LoadTexture(
      resources_.renderer, "assets/textures/ui/level_indicator.png");
  resources_.ui_resources.health_bar_texture =
      IMG_LoadTexture(resources_.renderer, "assets/textures/ui/health_bar.png");
  resources_.ui_resources.exp_bar_texture =
      IMG_LoadTexture(resources_.renderer, "assets/textures/ui/exp_bar.png");
  resources_.ui_resources.start_screen_texture = IMG_LoadTexture(
      resources_.renderer, "assets/textures/ui/start_screen.png");
  resources_.ui_resources.paused_texture =
      IMG_LoadTexture(resources_.renderer, "assets/textures/ui/paused.png");
  resources_.ui_resources.game_over_texture =
      IMG_LoadTexture(resources_.renderer, "assets/textures/ui/game_over.png");
  resources_.ui_resources.level_up_option_card_texture = IMG_LoadTexture(
      resources_.renderer, "assets/textures/ui/level_up_option.png");
  resources_.ui_resources.button_texture = IMG_LoadTexture(
      resources_.renderer, "assets/textures/ui/button_texture.png");
  resources_.ui_resources.digit_font_texture = IMG_LoadTexture(
      resources_.renderer, "assets/fonts/font_outlined_sprite_sheet.png");
  resources_.ui_resources.timer_hourglass_texture =
      IMG_LoadTexture(resources_.renderer, "assets/textures/hourglass.png");
  resources_.ui_resources.ui_font_medium =
      TTF_OpenFont("assets/fonts/november/novem___.ttf", kFontSizeMedium);
  resources_.ui_resources.ui_font_large =
      TTF_OpenFont("assets/fonts/november/novem___.ttf", kFontSizeLarge);

  if (resources_.ui_resources.ui_font_medium == nullptr ||
      resources_.ui_resources.ui_font_large == nullptr) {
    std::cerr << "TTF font could not be loaded: " << TTF_GetError()
              << std::endl;
    return false;
  }

  if (resources_.tile_texture == nullptr ||
      resources_.player_texture == nullptr ||
      resources_.enemy_texture == nullptr ||
      resources_.ui_resources.health_bar_texture == nullptr ||
      resources_.ui_resources.level_indicator_texture == nullptr ||
      resources_.ui_resources.exp_bar_texture == nullptr ||
      resources_.ui_resources.timer_hourglass_texture == nullptr ||
      resources_.ui_resources.game_over_texture == nullptr ||
      resources_.ui_resources.start_screen_texture == nullptr ||
      resources_.ui_resources.paused_texture == nullptr ||
      resources_.ui_resources.level_up_option_card_texture == nullptr ||
      resources_.ui_resources.button_texture == nullptr ||
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

bool RenderManager::InitializeCamera(const Player& player) {
  Vector2D player_centroid =
      GetCentroid(player.position_, player.stats_.sprite_size);
  camera_.position_.x = player_centroid.x - 0.5f * kWindowWidth;
  camera_.position_.y = player_centroid.y - 0.5f * kWindowHeight;

  return true;
};

void RenderManager::SetRenderColor(SDL_Renderer* renderer,
                                   const SDL_Color& color) {
  SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
}

void RenderManager::Render(const Scene& scene, float alpha, bool debug_mode,
                           float time, GameState game_state) {

  SetRenderColor(resources_.renderer, kColorBlack);
  SDL_RenderClear(resources_.renderer);

  if (game_state == in_start_screen) {
    RenderStartScreen();
  } else {

    UpdateCameraPosition(scene.player);

    camera_.render_position_ =
        LerpVector2D(camera_.prev_position_, camera_.position_, alpha);
    RenderTiledMap();
    RenderPlayer(scene.player, alpha);

    int num_enemy_vertices = SetupEnemyGeometry(scene.enemy, alpha);
    RenderEnemies(scene.enemy, num_enemy_vertices);
    SetupProjectileGeometry(scene.projectiles, alpha);
    RenderProjectiles(scene.projectiles);
    SetupGemGeometry(scene.exp_gem, alpha);
    RenderGem(scene.exp_gem);
    if (debug_mode) {
      RenderDebugWorldOccupancyMap(scene.occupancy_map);
      // RenderDebugEnemyOccupancyMap(scene.enemy, scene.occupancy_map, alpha);
      RenderDebugRayCaster(scene.enemy, alpha);
    };

    RenderUI(scene, time);
    if (game_state == is_gameover) {
      RenderGameOver();
    } else if (game_state == is_paused) {
      RenderPaused();
    } else if (game_state == in_level_up) {
      RenderLevelUp(scene.level_up_options);
    };
  }

  SDL_RenderPresent(resources_.renderer);
};

void RenderManager::UpdateCameraPosition(const Player& player) {
  Vector2D player_centroid =
      arelto::GetCentroid(player.position_, player.stats_.sprite_size);
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

void RenderManager::RenderTiledMap() {
  int top_left_tile_x =
      static_cast<int>(camera_.render_position_.x / kTileWidth);
  int top_left_tile_y =
      static_cast<int>(camera_.render_position_.y / kTileHeight);
  int bottom_right_tile_x = static_cast<int>(
      std::ceil((camera_.render_position_.x + kWindowWidth) / kTileWidth));
  int bottom_right_tile_y = static_cast<int>(
      std::ceil((camera_.render_position_.y + kWindowHeight) / kTileHeight));
  int start_x = std::max(0, top_left_tile_x);
  int end_x = std::min(kNumTilesX, bottom_right_tile_x);

  int start_y = std::max(0, top_left_tile_y);
  int end_y = std::min(kNumTilesY, bottom_right_tile_y);

  for (int i = start_x; i < end_x; ++i) {
    for (int j = start_y; j < end_y; ++j) {
      SDL_Rect render_rect = resources_.tile_manager.tiles_[i][j];
      render_rect.x -= static_cast<int>(camera_.render_position_.x);
      render_rect.y -= static_cast<int>(camera_.render_position_.y);
      int tile_id = resources_.tile_manager.tile_map_[i][j];
      const SDL_Rect& source_rect =
          resources_.tile_manager.select_tiles_[tile_id];
      SDL_RenderCopy(resources_.renderer, resources_.tile_texture, &source_rect,
                     &render_rect);
    }
  }
};

void RenderManager::RenderPlayer(const Player& player, float alpha) {
  Vector2D player_render_pos =
      LerpVector2D(player.prev_position_, player.position_, alpha);

  float x = player_render_pos.x - camera_.render_position_.x;
  float y = player_render_pos.y - camera_.render_position_.y;
  float w = static_cast<float>(player.stats_.sprite_size.width);
  float h = static_cast<float>(player.stats_.sprite_size.height);

  bool is_standing_still = player.velocity_.Norm() < 1e-3;
  bool is_facing_right = player.last_horizontal_velocity_ >= 0.0f;

  int src_x = ((SDL_GetTicks64() / kPlayerAnimationFrameDuration) %
               kPlayerNumSpriteCells) *
              kPlayerSpriteCellWidth;
  int src_y = is_standing_still ? 0 : kPlayerSpriteCellHeight;

  int texture_w, texture_h;
  SDL_QueryTexture(resources_.player_texture, nullptr, nullptr, &texture_w,
                   &texture_h);

  float u_left = static_cast<float>(src_x) / texture_w;
  float u_right =
      static_cast<float>(src_x + kPlayerSpriteCellWidth) / texture_w;
  float v_top = static_cast<float>(src_y) / texture_h;
  float v_bottom =
      static_cast<float>(src_y + kPlayerSpriteCellHeight) / texture_h;

  float vertex_left = is_facing_right ? u_left : u_right;
  float vertex_right = is_facing_right ? u_right : u_left;

  SDL_Color c = {255, 255, 255, 255};

  SDL_Vertex vertices[6] = {// Triangle 1 (Top-Left, Bottom-Left, Bottom-Right)
                            {{x, y}, c, {vertex_left, v_top}},
                            {{x, y + h}, c, {vertex_left, v_bottom}},
                            {{x + w, y + h}, c, {vertex_right, v_bottom}},
                            // Triangle 2 (Top-Left, Bottom-Right, Top-Right)
                            {{x, y}, c, {vertex_left, v_top}},
                            {{x + w, y + h}, c, {vertex_right, v_bottom}},
                            {{x + w, y}, c, {vertex_right, v_top}}};

  SDL_RenderGeometry(resources_.renderer, resources_.player_texture, vertices,
                     6, nullptr, 0);
};

int RenderManager::SetupEnemyGeometry(const Enemy& enemy, float alpha) {
  // The return type is int as we need to know how many vertices to actually
  // render when we call SDLRenderGeometry. So we traverse the enemies struct
  // and keep count of the total number of active vertices.

  int current_vertex_idx = 0;

  float cell_uv_width = 1.0f / (float)kEnemyNumSpriteCells;

  float cull_left = camera_.render_position_.x;
  float cull_right = camera_.render_position_.x + kWindowWidth;
  float cull_top = camera_.render_position_.y;
  float cull_bottom = camera_.render_position_.y + kWindowHeight;

  cull_left -= kRenderCullPadding;
  cull_right += kRenderCullPadding;
  cull_top -= kRenderCullPadding;
  cull_bottom += kRenderCullPadding;

  for (int i = 0; i < kNumEnemies; ++i) {
    if (!enemy.is_alive[i]) {
      continue;
    };

    float w = enemy.sprite_size[i].width;
    float h = enemy.sprite_size[i].height;

    // Skip setting up the enemy geometry if they are not in view.
    if (enemy.position[i].x + w < cull_left ||
        enemy.position[i].x > cull_right ||
        enemy.position[i].y + h < cull_top ||
        enemy.position[i].y > cull_bottom) {
      continue;
    }

    Vector2D render_enemy_pos =
        LerpVector2D(enemy.prev_position[i], enemy.position[i], alpha);

    float x = render_enemy_pos.x - camera_.render_position_.x;
    float y = render_enemy_pos.y - camera_.render_position_.y;

    uint16_t time_offset = i * 127;
    uint16_t frame_idx =
        ((SDL_GetTicks64() + time_offset) / kEnemyAnimationFrameDuration) %
        kEnemyNumSpriteCells;

    float u_left = frame_idx * cell_uv_width;
    float u_right = u_left + cell_uv_width;
    float v_top = kTexCoordTop;
    float v_bottom = kTexCoordBottom;

    bool is_facing_right = enemy.last_horizontal_velocity[i] > 0;

    float vertex_left = is_facing_right ? u_left : u_right;
    float vertex_right = is_facing_right ? u_right : u_left;

    // --- Vertices for Triangle 1 (Top-Left, Bottom-Left, Bottom-Right) ---
    // 1. Top-Left
    resources_.enemies_vertices_[current_vertex_idx + 0] = {
        {x, y}, {255, 255, 255, 255}, {vertex_left, v_top}};
    // 2. Bottom-Left
    resources_.enemies_vertices_[current_vertex_idx + 1] = {
        {x, y + h}, {255, 255, 255, 255}, {vertex_left, v_bottom}};
    // 3. Bottom-Right
    resources_.enemies_vertices_[current_vertex_idx + 2] = {
        {x + w, y + h}, {255, 255, 255, 255}, {vertex_right, v_bottom}};
    // --- Vertices for Triangle 2 (Top-Left, Bottom-Right, Top-Right) ---
    // 4. Top-Left (Repeat)
    resources_.enemies_vertices_[current_vertex_idx + 3] =
        resources_
            .enemies_vertices_[current_vertex_idx + 0];  // Same as vertex 1
    // 5. Bottom-Right (Repeat)
    resources_.enemies_vertices_[current_vertex_idx + 4] =
        resources_
            .enemies_vertices_[current_vertex_idx + 2];  // Same as vertex 3
    // 6. Top-Right
    resources_.enemies_vertices_[current_vertex_idx + 5] = {
        {x + w, y}, {255, 255, 255, 255}, {vertex_right, v_top}};

    current_vertex_idx += kEnemyVertices;
  }
  return current_vertex_idx;
};

void RenderManager::RenderEnemies(const Enemy& enemy, int num_vertices) {
  // We use the number of vertices calculated during the setup of the enemy
  // geometry to render the vertices.
  SDL_RenderGeometry(resources_.renderer, resources_.enemy_texture,
                     resources_.enemies_vertices_, num_vertices, nullptr, 0);
};

void RenderManager::SetupProjectileGeometry(const Projectiles& projectiles,
                                            float alpha) {
  resources_.projectile_vertices_grouped_.clear();
  size_t num_projectiles = projectiles.GetNumProjectiles();
  if (num_projectiles == 0) {
    return;
  }

  int current_vertex_idx = 0;
  float cell_uv_width = 1.0f / (float)kProjectileNumSpriteCells;

  float cull_left = camera_.render_position_.x;
  float cull_right = camera_.render_position_.x + kWindowWidth;
  float cull_top = camera_.render_position_.y;
  float cull_bottom = camera_.render_position_.y + kWindowHeight;

  cull_left -= kRenderCullPadding;
  cull_right += kRenderCullPadding;
  cull_top -= kRenderCullPadding;
  cull_bottom += kRenderCullPadding;

  for (int i = 0; i < num_projectiles; ++i) {
    float w = projectiles.sprite_size_[i].width;
    float h = projectiles.sprite_size_[i].height;

    // Skip setting up the projectile geometry if they are not in view.
    if (projectiles.position_[i].x + w < cull_left ||
        projectiles.position_[i].x > cull_right ||
        projectiles.position_[i].y + h < cull_top ||
        projectiles.position_[i].y > cull_bottom) {
      continue;
    }

    Vector2D proj_render_pos = LerpVector2D(projectiles.prev_position_[i],
                                            projectiles.position_[i], alpha);

    float x = proj_render_pos.x - camera_.render_position_.x;
    float y = proj_render_pos.y - camera_.render_position_.y;

    int texture_id = projectiles.proj_type_[i];

    uint16_t time_offset = i * 127;
    int frame_idx =
        ((SDL_GetTicks64() + time_offset) / kProjectileAnimationFrameDuration) %
        kProjectileNumSpriteCells;

    float u_left = frame_idx * cell_uv_width;
    float u_right = u_left + cell_uv_width;
    float v_top = kTexCoordTop;
    float v_bottom = kTexCoordBottom;

    bool is_facing_right = projectiles.direction_[i].x > 0;

    float vertex_left = is_facing_right ? u_left : u_right;
    float vertex_right = is_facing_right ? u_right : u_left;

    SDL_Vertex vertices[kProjectileVertices];

    // --- Vertices for Triangle 1 (Top-Left, Bottom-Left, Bottom-Right) ---
    // 1. Top-Left
    vertices[0] = {{x, y}, {255, 255, 255, 255}, {vertex_left, v_top}};
    // 2. Bottom-Left
    vertices[1] = {{x, y + h}, {255, 255, 255, 255}, {vertex_left, v_bottom}};
    // 3. Bottom-Right
    vertices[2] = {
        {x + w, y + h}, {255, 255, 255, 255}, {vertex_right, v_bottom}};
    // --- Vertices for Triangle 2 (Top-Left, Bottom-Right, Top-Right) ---
    // 4. Top-Left (Repeat)
    vertices[3] = vertices[0];  // Same as vertex 1
                                // 5. Bottom-Right (Repeat)
    vertices[4] = vertices[2];  // Same as vertex 3
    // 6. Top-Right
    vertices[5] = {{x + w, y}, {255, 255, 255, 255}, {vertex_right, v_top}};

    for (int j = 0; j < kProjectileVertices; ++j) {
      resources_.projectile_vertices_grouped_[texture_id].push_back(
          vertices[j]);
    }
  }
};

void RenderManager::RenderProjectiles(const Projectiles& projectiles) {
  for (const auto& pair : resources_.projectile_vertices_grouped_) {
    int texture_id = pair.first;
    const std::vector<SDL_Vertex>& vertices = pair.second;
    if (texture_id >= 0 && texture_id < resources_.projectile_textures.size()) {
      SDL_RenderGeometry(
          resources_.renderer, resources_.projectile_textures[texture_id],
          vertices.data(), static_cast<int>(vertices.size()), nullptr, 0);
    };
  };
};

void RenderManager::SetupGemGeometry(const ExpGem& exp_gem, float alpha) {
  resources_.gem_vertices_grouped_.clear();
  size_t num_gems = exp_gem.GetNumExpGems();
  if (num_gems == 0) {
    return;
  }

  int current_vertex_idx = 0;
  float cell_uv_width = 1.0f;

  float cull_left = camera_.render_position_.x;
  float cull_right = camera_.render_position_.x + kWindowWidth;
  float cull_top = camera_.render_position_.y;
  float cull_bottom = camera_.render_position_.y + kWindowHeight;

  cull_left -= kRenderCullPadding;
  cull_right += kRenderCullPadding;
  cull_top -= kRenderCullPadding;
  cull_bottom += kRenderCullPadding;

  for (int i = 0; i < num_gems; ++i) {
    float w = exp_gem.sprite_size_[i].width;
    float h = exp_gem.sprite_size_[i].height;

    // Skip setting up the projectile geometry if they are not in view.
    if (exp_gem.position_[i].x + w < cull_left ||
        exp_gem.position_[i].x > cull_right ||
        exp_gem.position_[i].y + h < cull_top ||
        exp_gem.position_[i].y > cull_bottom) {
      continue;
    }

    Vector2D gem_render_pos =
        LerpVector2D(exp_gem.prev_position_[i], exp_gem.position_[i], alpha);

    float x = gem_render_pos.x - camera_.render_position_.x;
    float y = gem_render_pos.y - camera_.render_position_.y;

    int texture_id = exp_gem.rarity_[i];

    int frame_idx = 0;

    float u_left = frame_idx * cell_uv_width;
    float u_right = u_left + cell_uv_width;
    float v_top = kTexCoordTop;
    float v_bottom = kTexCoordBottom;

    bool is_facing_right = true;

    float vertex_left = is_facing_right ? u_left : u_right;
    float vertex_right = is_facing_right ? u_right : u_left;

    SDL_Vertex vertices[kExpGemVertices];

    // --- Vertices for Triangle 1 (Top-Left, Bottom-Left, Bottom-Right) ---
    // 1. Top-Left
    vertices[0] = {{x, y}, {255, 255, 255, 255}, {vertex_left, v_top}};
    // 2. Bottom-Left
    vertices[1] = {{x, y + h}, {255, 255, 255, 255}, {vertex_left, v_bottom}};
    // 3. Bottom-Right
    vertices[2] = {
        {x + w, y + h}, {255, 255, 255, 255}, {vertex_right, v_bottom}};
    // --- Vertices for Triangle 2 (Top-Left, Bottom-Right, Top-Right) ---
    // 4. Top-Left (Repeat)
    vertices[3] = vertices[0];  // Same as vertex 1
                                // 5. Bottom-Right (Repeat)
    vertices[4] = vertices[2];  // Same as vertex 3
    // 6. Top-Right
    vertices[5] = {{x + w, y}, {255, 255, 255, 255}, {vertex_right, v_top}};

    for (int j = 0; j < kExpGemVertices; ++j) {
      resources_.gem_vertices_grouped_[texture_id].push_back(vertices[j]);
    }
  }
};

void RenderManager::RenderGem(const ExpGem& exp_gem) {
  for (const auto& pair : resources_.gem_vertices_grouped_) {
    int texture_id = pair.first;
    const std::vector<SDL_Vertex>& vertices = pair.second;
    if (texture_id >= 0 && texture_id < resources_.gem_textures.size()) {
      SDL_RenderGeometry(resources_.renderer,
                         resources_.gem_textures[texture_id], vertices.data(),
                         static_cast<int>(vertices.size()), nullptr, 0);
    };
  };
};

void RenderManager::RenderDebugWorldOccupancyMap(
    const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map) {
  // Get the original blend mode to be able to later restore it. The debug
  // visualization should blend textures, but regular rendering should not.
  SDL_BlendMode original_blend_mode;
  SDL_GetRenderDrawBlendMode(resources_.renderer, &original_blend_mode);
  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_BLEND);

  int grid_width_cells = kOccupancyMapWidth;
  int grid_height_cells = kOccupancyMapHeight;

  int top_left_x =
      static_cast<int>(camera_.render_position_.x / kOccupancyMapResolution);
  int top_left_y =
      static_cast<int>(camera_.render_position_.y / kOccupancyMapResolution);
  int bottom_right_x = static_cast<int>(std::ceil(
      (camera_.render_position_.x + kWindowWidth) / kOccupancyMapResolution));
  int bottom_right_y = static_cast<int>(std::ceil(
      (camera_.render_position_.y + kWindowHeight) / kOccupancyMapResolution));

  int start_x = std::max(0, top_left_x);
  int end_x = std::min(grid_width_cells, bottom_right_x);
  int start_y = std::max(0, top_left_y);
  int end_y = std::min(grid_height_cells, bottom_right_y);

  for (int i = start_x; i < end_x; ++i) {
    for (int j = start_y; j < end_y; ++j) {

      SDL_Rect render_rect;
      render_rect.x = static_cast<int>(i * kOccupancyMapResolution -
                                       camera_.render_position_.x);
      render_rect.y = static_cast<int>(j * kOccupancyMapResolution -
                                       camera_.render_position_.y);
      render_rect.w = kOccupancyMapResolution;
      render_rect.h = kOccupancyMapResolution;

      EntityType type = occupancy_map.Get(i, j);

      if (type != EntityType::None) {
        // Color coding based on type
        switch (type) {
          case EntityType::player:
            SetRenderColor(resources_.renderer, WithOpacity(kColorBlue, 128));
            break;
          case EntityType::enemy:
            SetRenderColor(resources_.renderer, WithOpacity(kColorRed, 128));
            break;
          case EntityType::projectile:
            SetRenderColor(resources_.renderer, WithOpacity(kColorYellow, 128));
            break;
          case EntityType::terrain:
            SetRenderColor(resources_.renderer, WithOpacity(kColorGreen, 128));
            break;
          default:
            SetRenderColor(resources_.renderer, WithOpacity(kColorGrey, 128));
            break;
        }
        // The rectangles are rendered first so the grid cells are on top.
        SDL_RenderFillRect(resources_.renderer, &render_rect);
      }

      SetRenderColor(resources_.renderer, WithOpacity(kColorBlack, 50));
      SDL_RenderDrawRect(resources_.renderer, &render_rect);
    }
  }

  SDL_SetRenderDrawBlendMode(resources_.renderer, original_blend_mode);
};

void RenderManager::RenderDebugRayCaster(const Enemy& enemy, float alpha) {
  SDL_BlendMode original_blend_mode;
  SDL_GetRenderDrawBlendMode(resources_.renderer, &original_blend_mode);
  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_BLEND);

  for (int i = 0; i < kNumEnemies; ++i) {
    if (!enemy.is_alive[i]) {
      continue;
    }

    Vector2D enemy_pos_world =
        LerpVector2D(enemy.prev_position[i], enemy.position[i], alpha);

    Vector2D center_world = enemy_pos_world + enemy.collider[i].offset;

    float half_w = enemy.collider[i].size.width * 0.5f;
    float half_h = enemy.collider[i].size.height * 0.5f;
    float ray_offset_dist = std::max(half_h, half_w) + kMinRayDistance;

    int ray_history_idx =
        (enemy.ray_caster.history_idx - 1 + kRayHistoryLength) %
        kRayHistoryLength;

    for (int k = 0; k < kNumRays; ++k) {
      float dist = enemy.ray_caster.ray_hit_distances[ray_history_idx][k][i];
      Vector2D dir = enemy.ray_caster.pattern.ray_dir[k];
      EntityType type = enemy.ray_caster.ray_hit_types[ray_history_idx][k][i];

      // The ray started 'ray_offset_dist' away from the center
      Vector2D ray_start_world = center_world + dir * ray_offset_dist;
      Vector2D ray_end_world = ray_start_world + dir * dist;

      Vector2D start_screen = ray_start_world - camera_.render_position_;
      Vector2D end_screen = ray_end_world - camera_.render_position_;

      switch (type) {
        case EntityType::player:
          SetRenderColor(resources_.renderer, WithOpacity(kColorRed, 150));
          break;
        case EntityType::terrain:
          SetRenderColor(resources_.renderer, WithOpacity(kColorGrey, 50));
          break;
        case EntityType::enemy:
          SetRenderColor(resources_.renderer, WithOpacity(kColorOrange, 150));
          break;
        default:
          SetRenderColor(resources_.renderer, WithOpacity(kColorBlack, 50));
          break;
      }

      SDL_RenderDrawLine(resources_.renderer, static_cast<int>(start_screen.x),
                         static_cast<int>(start_screen.y),
                         static_cast<int>(end_screen.x),
                         static_cast<int>(end_screen.y));
    }
  }

  SDL_SetRenderDrawBlendMode(resources_.renderer, original_blend_mode);
}
void RenderManager::RenderUI(const Scene& scene, float time) {
  ui_manager_.UpdateUI(scene, time);

  int group_x = static_cast<int>(ui_manager_.health_bar_.screen_position.x);
  int group_y = static_cast<int>(ui_manager_.health_bar_.screen_position.y);

  for (const auto& el : ui_manager_.health_bar_.elements) {
    SDL_Rect dst_rect;
    dst_rect.x = group_x + el.relative_offset.x;
    dst_rect.y = group_y + el.relative_offset.y;
    dst_rect.w = el.sprite_size.width;
    dst_rect.h = el.sprite_size.height;

    if (el.tag == UIElement::Tag::text) {
      RenderDigitString(el.text_value,
                        group_x + static_cast<int>(el.relative_offset.x),
                        group_y + static_cast<int>(el.relative_offset.y),
                        el.sprite_size, el.char_size);
      continue;
    }

    SDL_RenderCopy(resources_.renderer,
                   resources_.ui_resources.health_bar_texture, &el.src_rect,
                   &dst_rect);
  }
  group_x = static_cast<int>(ui_manager_.level_indicator_.screen_position.x);
  group_y = static_cast<int>(ui_manager_.level_indicator_.screen_position.y);

  for (const auto& el : ui_manager_.level_indicator_.elements) {
    SDL_Rect dst_rect;
    dst_rect.x = group_x + el.relative_offset.x;
    dst_rect.y = group_y + el.relative_offset.y;
    dst_rect.w = el.sprite_size.width;
    dst_rect.h = el.sprite_size.height;

    if (el.tag == UIElement::Tag::text) {
      RenderDigitString(el.text_value,
                        group_x + static_cast<int>(el.relative_offset.x),
                        group_y + static_cast<int>(el.relative_offset.y),
                        el.sprite_size, el.char_size);
      continue;
    }

    SDL_RenderCopy(resources_.renderer,
                   resources_.ui_resources.level_indicator_texture,
                   &el.src_rect, &dst_rect);
  }

  group_x = static_cast<int>(ui_manager_.exp_bar_.screen_position.x);
  group_y = static_cast<int>(ui_manager_.exp_bar_.screen_position.y);

  for (const auto& el : ui_manager_.exp_bar_.elements) {
    SDL_Rect dst_rect;
    dst_rect.x = group_x + el.relative_offset.x;
    dst_rect.y = group_y + el.relative_offset.y;
    dst_rect.w = el.sprite_size.width;
    dst_rect.h = el.sprite_size.height;

    if (el.tag == UIElement::Tag::text) {
      RenderDigitString(el.text_value,
                        group_x + static_cast<int>(el.relative_offset.x),
                        group_y + static_cast<int>(el.relative_offset.y),
                        el.sprite_size, el.char_size);
      continue;
    }

    SDL_RenderCopy(resources_.renderer, resources_.ui_resources.exp_bar_texture,
                   &el.src_rect, &dst_rect);
  }

  group_x = static_cast<int>(ui_manager_.timer_.screen_position.x);
  group_y = static_cast<int>(ui_manager_.timer_.screen_position.y);

  for (const auto& el : ui_manager_.timer_.elements) {
    SDL_Rect dst_rect;
    dst_rect.x = group_x + el.relative_offset.x;
    dst_rect.y = group_y + el.relative_offset.y;
    dst_rect.w = el.sprite_size.width;
    dst_rect.h = el.sprite_size.height;

    if (el.tag == UIElement::Tag::text) {
      RenderDigitString(el.text_value,
                        group_x + static_cast<int>(el.relative_offset.x),
                        group_y + static_cast<int>(el.relative_offset.y),
                        el.sprite_size, el.char_size);
      continue;
    }

    SDL_RenderCopy(resources_.renderer,
                   resources_.ui_resources.timer_hourglass_texture,
                   &el.src_rect, &dst_rect);
  };
};

void RenderManager::RenderDigitString(const std::string& text, int start_x,
                                      int start_y, Size2D sprite_size,
                                      Size2D char_size) {

  int char_width = char_size.width;
  int char_height = char_size.height;

  int current_x = start_x;

  for (char c : text) {
    SDL_Rect src_rect = {0, 0, static_cast<int>(sprite_size.width),
                         static_cast<int>(sprite_size.height)};

    if (c >= '0' && c <= '9') {
      int digit = c - '0';
      src_rect.x = digit * sprite_size.width;
    } else if (c == '/') {
      // TODO: Add more maintainable way of getting the sprite cell for a char.
      src_rect.x = sprite_size.width * 10;
    } else if (c == '-') {
      src_rect.x = sprite_size.width * 11;
    } else {
      // if not one of the above, they are not in the texture atlas, so we skip.
      current_x += char_width;
      continue;
    }

    SDL_Rect dest_rect = {current_x, start_y, char_width, char_height};

    SDL_RenderCopy(resources_.renderer,
                   resources_.ui_resources.digit_font_texture, &src_rect,
                   &dest_rect);

    current_x += char_width;
  };
};

void RenderManager::RenderStartScreen() {
  SDL_Rect dst_rect;
  dst_rect.x = 0;
  dst_rect.y = 0;
  dst_rect.w = kWindowWidth;
  dst_rect.h = kWindowHeight;

  SDL_RenderCopy(resources_.renderer,
                 resources_.ui_resources.start_screen_texture, nullptr,
                 &dst_rect);

  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_NONE);
};

void RenderManager::RenderGameOver() {

  SDL_Rect render_rect;
  render_rect.x = 0;
  render_rect.y = kWindowHeight / 3;
  render_rect.w = kWindowWidth;
  render_rect.h = kWindowHeight / 3;

  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_BLEND);
  SetRenderColor(resources_.renderer, WithOpacity(kColorBlack, 128));
  SDL_RenderFillRect(resources_.renderer, &render_rect);

  SDL_Rect dst_rect;
  dst_rect.x = (kWindowWidth - kGameOverSpriteWidth) / 2;
  dst_rect.y = (kWindowHeight - kGameOverSpriteHeight) / 2;
  dst_rect.w = kGameOverSpriteWidth;
  dst_rect.h = kGameOverSpriteHeight;

  SDL_RenderCopy(resources_.renderer, resources_.ui_resources.game_over_texture,
                 nullptr, &dst_rect);

  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_NONE);
};

void RenderManager::RenderPaused() {

  SDL_Rect render_rect;
  render_rect.x = 0;
  render_rect.y = kWindowHeight / 3;
  render_rect.w = kWindowWidth;
  render_rect.h = kWindowHeight / 3;

  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_BLEND);
  SetRenderColor(resources_.renderer, WithOpacity(kColorBlack, 128));
  SDL_RenderFillRect(resources_.renderer, &render_rect);

  SDL_Rect dst_rect;
  dst_rect.x = (kWindowWidth - kPausedSpriteWidth) / 2;
  dst_rect.y = (kWindowHeight - kPausedSpriteHeight) / 2;
  dst_rect.w = kPausedSpriteWidth;
  dst_rect.h = kPausedSpriteHeight;

  SDL_RenderCopy(resources_.renderer, resources_.ui_resources.paused_texture,
                 nullptr, &dst_rect);

  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_NONE);
};

void RenderManager::Shutdown() {

  if (resources_.player_texture) {
    SDL_DestroyTexture(resources_.player_texture);
    resources_.player_texture = nullptr;
  }

  if (resources_.enemy_texture) {
    SDL_DestroyTexture(resources_.enemy_texture);
    resources_.enemy_texture = nullptr;
  }

  if (resources_.ui_resources.digit_font_texture) {
    SDL_DestroyTexture(resources_.ui_resources.digit_font_texture);
    resources_.ui_resources.digit_font_texture = nullptr;
  }

  if (resources_.ui_resources.health_bar_texture) {
    SDL_DestroyTexture(resources_.ui_resources.health_bar_texture);
    resources_.ui_resources.health_bar_texture = nullptr;
  }

  if (resources_.ui_resources.timer_hourglass_texture) {
    SDL_DestroyTexture(resources_.ui_resources.timer_hourglass_texture);
    resources_.ui_resources.timer_hourglass_texture = nullptr;
  }

  if (resources_.ui_resources.level_up_option_card_texture) {
    SDL_DestroyTexture(resources_.ui_resources.level_up_option_card_texture);
    resources_.ui_resources.level_up_option_card_texture = nullptr;
  }

  if (resources_.ui_resources.button_texture) {
    SDL_DestroyTexture(resources_.ui_resources.button_texture);
    resources_.ui_resources.button_texture = nullptr;
  }

  IMG_Quit();

  if (resources_.ui_resources.ui_font_medium) {
    TTF_CloseFont(resources_.ui_resources.ui_font_medium);
    resources_.ui_resources.ui_font_medium = nullptr;
  }

  if (resources_.ui_resources.ui_font_large) {
    TTF_CloseFont(resources_.ui_resources.ui_font_large);
    resources_.ui_resources.ui_font_large = nullptr;
  }

  TTF_Quit();

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

void RenderManager::RenderLevelUp(
    const std::vector<std::unique_ptr<Upgrade>>& options) {
  SDL_Rect overlay_rect = {0, 0, kWindowWidth, kWindowHeight};
  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_BLEND);
  SetRenderColor(resources_.renderer, WithOpacity(kColorBlack, 150));
  SDL_RenderFillRect(resources_.renderer, &overlay_rect);

  int total_width = kNumUpgradeOptions * kLevelUpCardWidth +
                    (kNumUpgradeOptions - 1) * kLevelUpCardGap;
  int start_x = (kWindowWidth - total_width) / 2;
  int start_y = (kWindowHeight - kLevelUpCardHeight) / 2;

  for (size_t i = 0; i < options.size(); ++i) {
    const std::unique_ptr<Upgrade>& upgrade = options[i];

    int x =
        start_x + static_cast<int>(i * (kLevelUpCardWidth + kLevelUpCardGap));

    SDL_Rect card_rect = {x, start_y, kLevelUpCardWidth, kLevelUpCardHeight};

    SDL_RenderCopy(resources_.renderer,
                   resources_.ui_resources.level_up_option_card_texture,
                   nullptr, &card_rect);

    int spell_id = upgrade->GetSpellID();
    SDL_Texture* icon = resources_.projectile_textures[spell_id];
    SDL_Rect src_rect = {0, 0, kFireballSpriteWidth, kFireballSpriteHeight};
    SDL_Rect dest_rect = {x + (kLevelUpCardWidth - kLevelUpIconSize) / 2,
                          start_y + kLevelUpIconOffsetY, kLevelUpIconSize,
                          kLevelUpIconSize};
    SDL_RenderCopy(resources_.renderer, icon, &src_rect, &dest_rect);

    std::string spell_name = upgrade->GetSpellName();
    RenderText(spell_name, x, start_y + kLevelUpNameOffsetY, kColorWhite,
               resources_.ui_resources.ui_font_large, kLevelUpCardWidth);

    std::string description = upgrade->GetDescription();
    RenderText(description, x, start_y + kLevelUpDescOffsetY, kColorGrey,
               resources_.ui_resources.ui_font_medium, kLevelUpCardWidth);

    // "current_val -> new_val"
    std::string value_change =
        upgrade->GetOldValueString() + " -> " + upgrade->GetNewValueString();
    RenderText(value_change, x, start_y + kLevelUpStatsOffsetY, kColorGreen,
               resources_.ui_resources.ui_font_medium, kLevelUpCardWidth);

    int button_x = x + (kLevelUpCardWidth - kLevelUpButtonWidth) / 2;
    int button_y = start_y + kLevelUpButtonOffsetY;

    int mouse_x, mouse_y;
    SDL_GetMouseState(&mouse_x, &mouse_y);
    bool is_hovered =
        (mouse_x >= button_x && mouse_x <= button_x + kLevelUpButtonWidth &&
         mouse_y >= button_y && mouse_y <= button_y + kLevelUpButtonHeight);

    // button texture map has both default and hover state so y composant of top
    // left coord on the source rect depends on the hover state.
    SDL_Rect btn_src_rect = {
        0, is_hovered ? kLevelUpButtonTextureHeight / 2 : 0,
        kLevelUpButtonTextureWidth, kLevelUpButtonTextureHeight / 2};

    SDL_Rect btn_dst_rect = {button_x, button_y, kLevelUpButtonWidth,
                             kLevelUpButtonHeight};

    SDL_RenderCopy(resources_.renderer, resources_.ui_resources.button_texture,
                   &btn_src_rect, &btn_dst_rect);

    RenderText("SELECT", button_x, button_y + (kLevelUpButtonHeight - 26) / 2,
               kColorWhite, resources_.ui_resources.ui_font_medium,
               kLevelUpButtonWidth);
  }

  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_NONE);
}

// Render a string of text at a specified location (x,y) with a given color
// and font.
// Optional: If you specify a positive center_width, then it will center-align
// the text along that center_width.
void RenderManager::RenderText(const std::string& text, int x, int y,
                               SDL_Color color, TTF_Font* font,
                               int center_width) {
  SDL_Surface* surface = TTF_RenderText_Blended(font, text.c_str(), color);
  SDL_Texture* texture =
      SDL_CreateTextureFromSurface(resources_.renderer, surface);

  int render_x = x;
  if (center_width > 0) {
    render_x = x + (center_width - surface->w) / 2;
  }

  SDL_Rect dest = {render_x, y, surface->w, surface->h};
  SDL_RenderCopy(resources_.renderer, texture, nullptr, &dest);

  SDL_DestroyTexture(texture);
  SDL_FreeSurface(surface);
}

}  // namespace arelto
