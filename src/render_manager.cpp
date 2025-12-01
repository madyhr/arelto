// src/render_manager.cpp
#include "render_manager.h"
#include <SDL.h>
#include <SDL_image.h>
#include <SDL_timer.h>
#include <algorithm>
#include <iostream>
#include "entity.h"
#include "types.h"

namespace rl2 {

RenderManager::RenderManager(){};
RenderManager::~RenderManager(){
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

bool RenderManager::InitializeCamera(const Player& player) {
  Vector2D player_centroid =
      GetCentroid(player.position_, player.stats_.size);
  camera_.position_.x = player_centroid.x - 0.5f * kWindowWidth;
  camera_.position_.y = player_centroid.y - 0.5f * kWindowHeight;

  return true;
};



void RenderManager::Render(
    const Player& player, const Enemy& enemy, const Projectiles& projectiles,
    const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map,
    float alpha, bool debug_mode) {

  UpdateCameraPosition(player);

  camera_.render_position_ =
      LerpVector2D(camera_.prev_position_, camera_.position_, alpha);
  SDL_SetRenderDrawColor(resources_.renderer, 0x00, 0x00, 0x00, 0xFF);
  SDL_RenderClear(resources_.renderer);
  RenderTiledMap();
  RenderPlayer(player, alpha);

  int num_enemy_vertices = SetupEnemyGeometry(enemy, alpha);
  RenderEnemies(enemy, num_enemy_vertices);
  SetupProjectileGeometry(projectiles, alpha);
  RenderProjectiles(projectiles);
  if (debug_mode) {
    // RenderDebugWorldOccupancyMap(occupancy_map);
    RenderDebugEnemyOccupancyMap(enemy, occupancy_map, alpha);
  };

  SDL_RenderPresent(resources_.renderer);
};

void RenderManager::UpdateCameraPosition(const Player& player) {
  Vector2D player_centroid =
      rl2::GetCentroid(player.position_, player.stats_.size);
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
  int top_left_tile_x = static_cast<int>(camera_.position_.x / kTileWidth);
  int top_left_tile_y = static_cast<int>(camera_.position_.y / kTileHeight);
  int bottom_right_tile_x = static_cast<int>(
      std::ceil((camera_.position_.x + kWindowWidth) / kTileWidth));
  int bottom_right_tile_y = static_cast<int>(
      std::ceil((camera_.position_.y + kWindowHeight) / kTileHeight));
  int start_x = std::max(0, top_left_tile_x);
  int end_x = std::min(kNumTilesX, bottom_right_tile_x);

  int start_y = std::max(0, top_left_tile_y);
  int end_y = std::min(kNumTilesY, bottom_right_tile_y);

  for (int i = start_x; i < end_x; ++i) {
    for (int j = start_y; j < end_y; ++j) {
      SDL_Rect render_rect = resources_.tile_manager.tiles_[i][j];
      render_rect.x -= static_cast<int>(camera_.position_.x);
      render_rect.y -= static_cast<int>(camera_.position_.y);
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
  SDL_Rect player_render_box = {
      static_cast<int>(player_render_pos.x - camera_.position_.x),
      static_cast<int>(player_render_pos.y - camera_.position_.y),
      static_cast<int>(player.stats_.size.width),
      static_cast<int>(player.stats_.size.height)};

  bool is_standing_still = player.velocity_.Norm() < 1e-3;
  bool is_facing_right = player.last_horizontal_velocity_ >= 0.0f;

  SDL_Rect src_rect;
  src_rect.w = kPlayerSpriteCellWidth;
  src_rect.h = kPlayerSpriteCellHeight;
  src_rect.y = is_standing_still ? 0 : kPlayerSpriteCellHeight;
  src_rect.x = ((SDL_GetTicks64() / kPlayerAnimationFrameDuration) %
                kPlayerNumSpriteCells) *
               src_rect.w;

  SDL_RendererFlip is_flipped =
      is_facing_right ? SDL_FLIP_NONE : SDL_FLIP_HORIZONTAL;

  SDL_RenderCopyEx(resources_.renderer, resources_.player_texture, &src_rect,
                   &player_render_box, 0.0, nullptr, is_flipped);

};

int RenderManager::SetupEnemyGeometry(const Enemy& enemy, float alpha) {
  // The return type is int as we need to know how many vertices to actually
  // render when we call SDLRenderGeometry. So we traverse the enemies struct
  // and keep count of the total number of active vertices.

  int current_vertex_idx = 0;

  float cell_uv_width = 1.0f / (float)kEnemyNumSpriteCells;

  float cull_left = camera_.position_.x;
  float cull_right = camera_.position_.x + kWindowWidth;
  float cull_top = camera_.position_.y;
  float cull_bottom = camera_.position_.y + kWindowHeight;

  cull_left -= kCullPadding;
  cull_right += kCullPadding;
  cull_top -= kCullPadding;
  cull_bottom += kCullPadding;

  for (int i = 0; i < kNumEnemies; ++i) {
    if (!enemy.is_alive[i]) {
      continue;
    };

    float w = enemy.size[i].width;
    float h = enemy.size[i].height;

    // Skip setting up the enemy geometry if they are not in view.
    if (enemy.position[i].x + w < cull_left ||
        enemy.position[i].x > cull_right ||
        enemy.position[i].y + h < cull_top ||
        enemy.position[i].y > cull_bottom) {
      continue;
    }

    Vector2D render_enemy_pos =
        LerpVector2D(enemy.prev_position[i], enemy.position[i], alpha);

    float x = render_enemy_pos.x - camera_.position_.x;
    float y = render_enemy_pos.y - camera_.position_.y;

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

  float cull_left = camera_.position_.x;
  float cull_right = camera_.position_.x + kWindowWidth;
  float cull_top = camera_.position_.y;
  float cull_bottom = camera_.position_.y + kWindowHeight;

  cull_left -= kCullPadding;
  cull_right += kCullPadding;
  cull_top -= kCullPadding;
  cull_bottom += kCullPadding;

  for (int i = 0; i < num_projectiles; ++i) {
    float w = projectiles.size_[i].width;
    float h = projectiles.size_[i].height;

    // Skip setting up the projectile geometry if they are not in view.
    if (projectiles.position_[i].x + w < cull_left ||
        projectiles.position_[i].x > cull_right ||
        projectiles.position_[i].y + h < cull_top ||
        projectiles.position_[i].y > cull_bottom) {
      continue;
    }

    Vector2D proj_render_pos = LerpVector2D(projectiles.prev_position_[i],
                                            projectiles.position_[i], alpha);

    float x = proj_render_pos.x - camera_.position_.x;
    float y = proj_render_pos.y - camera_.position_.y;

    int texture_id = projectiles.proj_id_[i];

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

void RenderManager::RenderDebugWorldOccupancyMap(
    const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map) {
  SDL_BlendMode original_blend_mode;
  SDL_GetRenderDrawBlendMode(resources_.renderer, &original_blend_mode);
  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_BLEND);

  int grid_width_cells = kMapWidth / kOccupancyMapResolution;
  int grid_height_cells = kMapHeight / kOccupancyMapResolution;

  int top_left_x =
      static_cast<int>(camera_.position_.x / kOccupancyMapResolution);
  int top_left_y =
      static_cast<int>(camera_.position_.y / kOccupancyMapResolution);
  int bottom_right_x = static_cast<int>(std::ceil(
      (camera_.position_.x + kWindowWidth) / kOccupancyMapResolution));
  int bottom_right_y = static_cast<int>(std::ceil(
      (camera_.position_.y + kWindowHeight) / kOccupancyMapResolution));

  int start_x = std::max(0, top_left_x);
  int end_x = std::min(grid_width_cells, bottom_right_x);
  int start_y = std::max(0, top_left_y);
  int end_y = std::min(grid_height_cells, bottom_right_y);

  for (int i = start_x; i < end_x; ++i) {
    for (int j = start_y; j < end_y; ++j) {

      SDL_Rect render_rect;
      render_rect.x =
          static_cast<int>(i * kOccupancyMapResolution - camera_.position_.x);
      render_rect.y =
          static_cast<int>(j * kOccupancyMapResolution - camera_.position_.y);
      render_rect.w = kOccupancyMapResolution;
      render_rect.h = kOccupancyMapResolution;

      EntityType type = occupancy_map.Get(i, j);

      if (type != EntityType::None) {
        // Color coding based on type
        switch (type) {
          case EntityType::player:
            SDL_SetRenderDrawColor(resources_.renderer, 0, 0, 255,
                                   128);  // Blue
            break;
          case EntityType::enemy:
            SDL_SetRenderDrawColor(resources_.renderer, 255, 0, 0, 128);  // Red
            break;
          case EntityType::projectile:
            SDL_SetRenderDrawColor(resources_.renderer, 255, 255, 0,
                                   128);  // Yellow
            break;
          case EntityType::terrain:
            SDL_SetRenderDrawColor(resources_.renderer, 0, 255, 0,
                                   128);  // Green
            break;
          default:
            SDL_SetRenderDrawColor(resources_.renderer, 100, 100, 100,
                                   128);  // Grey (Generic)
            break;
        }
        // The rectangles are rendered first so the grid cells are on top.
        SDL_RenderFillRect(resources_.renderer, &render_rect);
      }

      SDL_SetRenderDrawColor(resources_.renderer, 0, 0, 0, 50);
      SDL_RenderDrawRect(resources_.renderer, &render_rect);
    }
  }

  // Restore original blend mode
  SDL_SetRenderDrawBlendMode(resources_.renderer, original_blend_mode);
};

void RenderManager::RenderDebugEnemyOccupancyMap(
    const Enemy& enemy,
    const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map,
    float alpha) {
  SDL_BlendMode originalBlendMode;
  SDL_GetRenderDrawBlendMode(resources_.renderer, &originalBlendMode);
  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_BLEND);

  const int half_w = static_cast<int>(kEnemyOccupancyMapWidth / 2);
  const int half_h = static_cast<int>(kEnemyOccupancyMapHeight / 2);

  for (int i = 0; i < kNumEnemies; ++i) {
    if (!enemy.is_alive[i]) {
      continue;
    }

    Vector2D enemy_render_pos =
        LerpVector2D(enemy.prev_position[i], enemy.position[i], alpha);

    // Vector2D enemy_grid_pos = WorldToGrid(enemy.position[i]);
    Vector2D enemy_grid_pos =
        WorldToGrid(GetCentroid(enemy_render_pos, enemy.size[i]));
    int start_world_x = static_cast<int>(enemy_grid_pos.x) - half_w;
    int start_world_y = static_cast<int>(enemy_grid_pos.y) - half_h;

    // Draw outline of the enemy's vision
    SDL_Rect vision_rect;
    vision_rect.x = static_cast<int>(start_world_x * kOccupancyMapResolution -
                                     camera_.position_.x);
    vision_rect.y = static_cast<int>(start_world_y * kOccupancyMapResolution -
                                     camera_.position_.y);
    vision_rect.w = kEnemyOccupancyMapWidth * kOccupancyMapResolution;
    vision_rect.h = kEnemyOccupancyMapHeight * kOccupancyMapResolution;

    SDL_SetRenderDrawColor(resources_.renderer, 255, 255, 255, 200);
    SDL_RenderDrawRect(resources_.renderer, &vision_rect);

    for (int local_y = 0; local_y < kEnemyOccupancyMapHeight; ++local_y) {
      for (int local_x = 0; local_x < kEnemyOccupancyMapWidth; ++local_x) {
        EntityType type = enemy.occupancy_map[i].Get(local_x, local_y);

        if (type != EntityType::None) {
          int world_x = start_world_x + local_x;
          int world_y = start_world_y + local_y;

          SDL_Rect render_rect;
          render_rect.x = static_cast<int>(world_x * kOccupancyMapResolution -
                                           camera_.position_.x);
          render_rect.y = static_cast<int>(world_y * kOccupancyMapResolution -
                                           camera_.position_.y);
          render_rect.w = kOccupancyMapResolution;
          render_rect.h = kOccupancyMapResolution;

          switch (type) {
            case EntityType::player:
              SDL_SetRenderDrawColor(resources_.renderer, 0, 0, 255, 128);
              break;
            case EntityType::enemy:
              SDL_SetRenderDrawColor(resources_.renderer, 255, 0, 0, 128);
              break;
            case EntityType::projectile:
              SDL_SetRenderDrawColor(resources_.renderer, 255, 255, 0, 128);
              break;
            case EntityType::terrain:
              SDL_SetRenderDrawColor(resources_.renderer, 0, 255, 0, 128);
              break;
            default:
              SDL_SetRenderDrawColor(resources_.renderer, 100, 100, 100, 128);
              break;
          }
          SDL_RenderFillRect(resources_.renderer, &render_rect);

          SDL_SetRenderDrawColor(resources_.renderer, 255, 255, 255, 50);
          SDL_RenderDrawRect(resources_.renderer, &render_rect);
        }
      }
    }
  }

  SDL_SetRenderDrawBlendMode(resources_.renderer, originalBlendMode);
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
