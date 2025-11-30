// src/game_render.cpp
#include "game.h"

void rl2::Game::Render(float alpha) {
  if (game_status_.in_headless_mode) {
    return;
  }
  // Setting alpha to 1.0f to always render the latest state.
  Game::GenerateOutput(alpha);
}

void rl2::Game::GenerateOutput(float alpha) {

  Game::UpdateCameraPosition();

  Vector2D render_cam_pos =
      LerpVector2D(camera_.prev_position_, camera_.position_, alpha);
  SDL_SetRenderDrawColor(resources_.renderer, 0x00, 0x00, 0x00, 0xFF);
  SDL_RenderClear(resources_.renderer);
  RenderTiledMap(render_cam_pos);
  RenderPlayer(alpha, render_cam_pos);

  int num_enemy_vertices = SetupEnemyGeometry(alpha, render_cam_pos);
  RenderEnemies(num_enemy_vertices);

  SetupProjectileGeometry(alpha, render_cam_pos);
  RenderProjectiles();
  if (game_status_.in_debug_mode) {
    // RenderDebugWorldOccupancyMap();
    RenderDebugEnemyOccupancyMap(alpha, render_cam_pos);
  };

  SDL_RenderPresent(resources_.renderer);
};

void rl2::Game::UpdateCameraPosition() {
  Vector2D player_centroid =
      rl2::GetCentroid(player_.position_, player_.stats_.size);
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

void rl2::Game::RenderTiledMap(Vector2D cam_pos) {
  int top_left_tile_x = static_cast<int>(cam_pos.x / kTileWidth);
  int top_left_tile_y = static_cast<int>(cam_pos.y / kTileHeight);
  int bottom_right_tile_x =
      static_cast<int>(std::ceil((cam_pos.x + kWindowWidth) / kTileWidth));
  int bottom_right_tile_y =
      static_cast<int>(std::ceil((cam_pos.y + kWindowHeight) / kTileHeight));
  int start_x = std::max(0, top_left_tile_x);
  int end_x = std::min(kNumTilesX, bottom_right_tile_x);

  int start_y = std::max(0, top_left_tile_y);
  int end_y = std::min(kNumTilesY, bottom_right_tile_y);

  for (int i = start_x; i < end_x; ++i) {
    for (int j = start_y; j < end_y; ++j) {
      SDL_Rect render_rect = resources_.tile_manager.tiles_[i][j];
      render_rect.x -= static_cast<int>(cam_pos.x);
      render_rect.y -= static_cast<int>(cam_pos.y);
      int tile_id = resources_.tile_manager.tile_map_[i][j];
      const SDL_Rect& source_rect =
          resources_.tile_manager.select_tiles_[tile_id];
      SDL_RenderCopy(resources_.renderer, resources_.tile_texture, &source_rect,
                     &render_rect);
    }
  }
};

void rl2::Game::RenderPlayer(float alpha, Vector2D cam_pos) {

  Vector2D player_render_pos =
      LerpVector2D(player_.prev_position_, player_.position_, alpha);
  SDL_Rect player_render_box = {
      static_cast<int>(player_render_pos.x - cam_pos.x),
      static_cast<int>(player_render_pos.y - cam_pos.y),
      static_cast<int>(player_.stats_.size.width),
      static_cast<int>(player_.stats_.size.height)};

  bool is_standing_still = player_.velocity_.Norm() < 1e-3;
  bool is_facing_right = player_.last_horizontal_velocity_ >= 0.0f;

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

  if (player_.velocity_.x != 0) {
    player_.last_horizontal_velocity_ = player_.velocity_.x;
  }
};

int rl2::Game::SetupEnemyGeometry(float alpha, Vector2D cam_pos) {
  // The return type is int as we need to know how many vertices to actually
  // render when we call SDLRenderGeometry. So we traverse the enemies struct
  // and keep count of the total number of active vertices.

  int current_vertex_idx = 0;

  float cell_uv_width = 1.0f / (float)kEnemyNumSpriteCells;

  float cull_left = cam_pos.x;
  float cull_right = cam_pos.x + kWindowWidth;
  float cull_top = cam_pos.y;
  float cull_bottom = cam_pos.y + kWindowHeight;

  cull_left -= kCullPadding;
  cull_right += kCullPadding;
  cull_top -= kCullPadding;
  cull_bottom += kCullPadding;

  for (int i = 0; i < kNumEnemies; ++i) {
    if (!enemy_.is_alive[i]) {
      continue;
    };

    float w = enemy_.size[i].width;
    float h = enemy_.size[i].height;

    // Skip setting up the enemy geometry if they are not in view.
    if (enemy_.position[i].x + w < cull_left ||
        enemy_.position[i].x > cull_right ||
        enemy_.position[i].y + h < cull_top ||
        enemy_.position[i].y > cull_bottom) {
      continue;
    }

    Vector2D render_enemy_pos =
        LerpVector2D(enemy_.prev_position[i], enemy_.position[i], alpha);

    float x = render_enemy_pos.x - cam_pos.x;
    float y = render_enemy_pos.y - cam_pos.y;

    uint16_t time_offset = i * 127;
    uint16_t frame_idx =
        ((SDL_GetTicks64() + time_offset) / kEnemyAnimationFrameDuration) %
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

void rl2::Game::RenderEnemies(int num_vertices) {
  // We use the number of vertices calculated during the setup of the enemy
  // geometry to render the vertices.
  SDL_RenderGeometry(resources_.renderer, resources_.enemy_texture,
                     enemies_vertices_, num_vertices, nullptr, 0);
};

void rl2::Game::SetupProjectileGeometry(float alpha, Vector2D cam_pos) {
  projectile_vertices_grouped_.clear();
  size_t num_projectiles = projectiles_.GetNumProjectiles();
  if (num_projectiles == 0) {
    return;
  }

  int current_vertex_idx = 0;
  float cell_uv_width = 1.0f / (float)kProjectileNumSpriteCells;

  float cull_left = cam_pos.x;
  float cull_right = cam_pos.x + kWindowWidth;
  float cull_top = cam_pos.y;
  float cull_bottom = cam_pos.y + kWindowHeight;

  cull_left -= kCullPadding;
  cull_right += kCullPadding;
  cull_top -= kCullPadding;
  cull_bottom += kCullPadding;

  for (int i = 0; i < num_projectiles; ++i) {
    float w = projectiles_.size_[i].width;
    float h = projectiles_.size_[i].height;

    // Skip setting up the projectile geometry if they are not in view.
    if (projectiles_.position_[i].x + w < cull_left ||
        projectiles_.position_[i].x > cull_right ||
        projectiles_.position_[i].y + h < cull_top ||
        projectiles_.position_[i].y > cull_bottom) {
      continue;
    }

    Vector2D proj_render_pos = LerpVector2D(projectiles_.prev_position_[i],
                                            projectiles_.position_[i], alpha);

    float x = proj_render_pos.x - cam_pos.x;
    float y = proj_render_pos.y - cam_pos.y;

    int texture_id = projectiles_.proj_id_[i];

    uint16_t time_offset = i * 127;
    int frame_idx =
        ((SDL_GetTicks64() + time_offset) / kProjectileAnimationFrameDuration) %
        kProjectileNumSpriteCells;

    float u_left = frame_idx * cell_uv_width;
    float u_right = u_left + cell_uv_width;
    float v_top = kTexCoordTop;
    float v_bottom = kTexCoordBottom;

    bool is_facing_right = projectiles_.direction_[i].x > 0;

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
      projectile_vertices_grouped_[texture_id].push_back(vertices[j]);
    }
  }
};

void rl2::Game::RenderProjectiles() {

  for (const auto& pair : projectile_vertices_grouped_) {
    int texture_id = pair.first;
    const std::vector<SDL_Vertex>& vertices = pair.second;
    if (texture_id >= 0 && texture_id < resources_.projectile_textures.size()) {
      SDL_RenderGeometry(
          resources_.renderer, resources_.projectile_textures[texture_id],
          vertices.data(), static_cast<int>(vertices.size()), nullptr, 0);
    };
  };
};

void rl2::Game::RenderDebugWorldOccupancyMap(Vector2D cam_pos) {
  SDL_BlendMode original_blend_mode;
  SDL_GetRenderDrawBlendMode(resources_.renderer, &original_blend_mode);
  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_BLEND);

  int grid_width_cells = kMapWidth / kOccupancyMapResolution;
  int grid_height_cells = kMapHeight / kOccupancyMapResolution;

  int top_left_x = static_cast<int>(cam_pos.x / kOccupancyMapResolution);
  int top_left_y = static_cast<int>(cam_pos.y / kOccupancyMapResolution);
  int bottom_right_x = static_cast<int>(
      std::ceil((cam_pos.x + kWindowWidth) / kOccupancyMapResolution));
  int bottom_right_y = static_cast<int>(
      std::ceil((cam_pos.y + kWindowHeight) / kOccupancyMapResolution));

  int start_x = std::max(0, top_left_x);
  int end_x = std::min(grid_width_cells, bottom_right_x);
  int start_y = std::max(0, top_left_y);
  int end_y = std::min(grid_height_cells, bottom_right_y);

  for (int i = start_x; i < end_x; ++i) {
    for (int j = start_y; j < end_y; ++j) {

      SDL_Rect render_rect;
      render_rect.x = static_cast<int>(i * kOccupancyMapResolution - cam_pos.x);
      render_rect.y = static_cast<int>(j * kOccupancyMapResolution - cam_pos.y);
      render_rect.w = kOccupancyMapResolution;
      render_rect.h = kOccupancyMapResolution;

      EntityType type = world_occupancy_map_.Get(i, j);

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

void rl2::Game::RenderDebugEnemyOccupancyMap(float alpha, Vector2D cam_pos) {
  SDL_BlendMode originalBlendMode;
  SDL_GetRenderDrawBlendMode(resources_.renderer, &originalBlendMode);
  SDL_SetRenderDrawBlendMode(resources_.renderer, SDL_BLENDMODE_BLEND);

  const int half_w = static_cast<int>(kEnemyOccupancyMapWidth / 2);
  const int half_h = static_cast<int>(kEnemyOccupancyMapHeight / 2);

  for (int i = 0; i < kNumEnemies; ++i) {
    if (!enemy_.is_alive[i]) {
      continue;
    }

    Vector2D enemy_render_pos =
        LerpVector2D(enemy_.prev_position[i], enemy_.position[i], alpha);

    // Vector2D enemy_grid_pos = WorldToGrid(enemy_.position[i]);
    Vector2D enemy_grid_pos =
        WorldToGrid(GetCentroid(enemy_render_pos, enemy_.size[i]));
    int start_world_x = static_cast<int>(enemy_grid_pos.x) - half_w;
    int start_world_y = static_cast<int>(enemy_grid_pos.y) - half_h;

    // Draw outline of the enemy's vision
    SDL_Rect vision_rect;
    vision_rect.x =
        static_cast<int>(start_world_x * kOccupancyMapResolution - cam_pos.x);
    vision_rect.y =
        static_cast<int>(start_world_y * kOccupancyMapResolution - cam_pos.y);
    vision_rect.w = kEnemyOccupancyMapWidth * kOccupancyMapResolution;
    vision_rect.h = kEnemyOccupancyMapHeight * kOccupancyMapResolution;

    SDL_SetRenderDrawColor(resources_.renderer, 255, 255, 255, 200);
    SDL_RenderDrawRect(resources_.renderer, &vision_rect);

    for (int local_y = 0; local_y < kEnemyOccupancyMapHeight; ++local_y) {
      for (int local_x = 0; local_x < kEnemyOccupancyMapWidth; ++local_x) {
        EntityType type = enemy_.occupancy_map[i].Get(local_x, local_y);

        if (type != EntityType::None) {
          int world_x = start_world_x + local_x;
          int world_y = start_world_y + local_y;

          SDL_Rect render_rect;
          render_rect.x =
              static_cast<int>(world_x * kOccupancyMapResolution - cam_pos.x);
          render_rect.y =
              static_cast<int>(world_y * kOccupancyMapResolution - cam_pos.y);
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

