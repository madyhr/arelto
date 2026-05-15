// src/render_manager.cpp
#include "render_manager.h"
#include <SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_timer.h>
#include <algorithm>
#include <iostream>
#include <map>
#include "config/ui_config_yaml.h"  // IWYU pragma: keep
#include "constants/chest.h"
#include "constants/enemy.h"
#include "constants/exp_gem.h"
#include "constants/game.h"
#include "constants/map.h"
#include "constants/player.h"
#include "constants/projectile.h"
#include "constants/ray_caster.h"
#include "constants/render.h"
#include "entity.h"
#include "scene.h"
#include "types.h"
#include "ui/containers.h"
#include "ui/widgets.h"
#include "yaml-cpp/yaml.h"

namespace arelto {

RenderManager::RenderManager() {};
RenderManager::~RenderManager() {};

bool RenderManager::Initialize(
    bool is_headless, EventManager& event_manager,
    const SpellTextureMapping& spell_texture_mapping) {

  if (is_headless) {
    return true;
  }

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL could not initialize! SDL Error: " << SDL_GetError()
              << '\n';
    return false;
  }

  window_ =
      SDL_CreateWindow("RL2", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                       kWindowWidth, kWindowHeight, SDL_WINDOW_SHOWN);
  if (!window_) {
    std::cerr << "Window could not be created: " << SDL_GetError() << '\n';
    return false;
  }

  renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);
  if (!renderer_) {
    std::cerr << "Renderer could not be created: " << SDL_GetError() << '\n';
    return false;
  }

  int img_flags = IMG_INIT_PNG;
  if (!(IMG_Init(img_flags) & img_flags)) {
    std::cerr << "SDL Images could not be initialized: " << SDL_GetError()
              << '\n';
    return false;
  }

  if (TTF_Init() == -1) {
    std::cerr << "SDL_ttf could not be initialized: " << TTF_GetError() << '\n';
    return false;
  }

  YAML::Node manifest;
  try {
    manifest = YAML::LoadFile("assets/config/textures.yaml");
  } catch (const YAML::Exception& e) {
    std::cerr << "Failed to load texture manifest: " << e.what() << '\n';
    return false;
  }

  if (!LoadTextures(manifest, spell_texture_mapping))
    return false;

  if (!LoadFonts(manifest))
    return false;

  if (!ValidateTextures())
    return false;

  tile_manager_.SetupTileMap();
  tile_manager_.SetupTiles();
  tile_manager_.SetupTileSelector();

  LoadUIConfig();

  ui_manager_.SetupUI(resources_, ui_config_, event_manager);

  return true;
}

SDL_Texture* RenderManager::LoadTexture(const std::string& section,
                                        const std::string& key,
                                        const YAML::Node& manifest) {
  std::string full_key = section + "." + key;
  if (!manifest[section] || !manifest[section][key]) {
    std::cerr << "Missing texture in manifest: " << full_key << '\n';
    return nullptr;
  }
  std::string path = manifest[section][key].as<std::string>();
  SDL_Texture* tex = IMG_LoadTexture(renderer_, path.c_str());
  if (!tex) {
    std::cerr << "Failed to load texture: " << path << '\n';
    return nullptr;
  }
  return tex;
}

bool RenderManager::LoadTextures(
    const YAML::Node& manifest,
    const SpellTextureMapping& spell_texture_mapping) {
  resources_.tile = LoadTexture("game", "tile", manifest);
  resources_.player = LoadTexture("game", "player", manifest);
  resources_.enemy = LoadTexture("game", "enemy", manifest);

  resources_.gems.push_back(LoadTexture("game", "gem_common", manifest));
  resources_.gems.push_back(LoadTexture("game", "gem_rare", manifest));
  resources_.gems.push_back(LoadTexture("game", "gem_epic", manifest));
  resources_.gems.push_back(LoadTexture("game", "gem_legendary", manifest));

  resources_.chest = LoadTexture("game", "chest", manifest);

  resources_.items.push_back(LoadTexture("items", "elia_armor", manifest));
  resources_.items.push_back(LoadTexture("items", "damodei_claw", manifest));
  resources_.items.push_back(LoadTexture("items", "volmnih_boots", manifest));
  resources_.items.push_back(
      LoadTexture("items", "sarto_button_bible", manifest));
  resources_.items.push_back(LoadTexture("items", "aiayn_scale", manifest));

  resources_.digit_font = LoadTexture("ui", "digit_font", manifest);
  resources_.level_indicator = LoadTexture("ui", "level_indicator", manifest);
  resources_.health_bar = LoadTexture("ui", "health_bar", manifest);
  resources_.exp_bar = LoadTexture("ui", "exp_bar", manifest);
  resources_.start_screen = LoadTexture("ui", "start_screen", manifest);
  resources_.game_over = LoadTexture("ui", "game_over", manifest);
  resources_.level_up_option_card =
      LoadTexture("ui", "level_up_option", manifest);
  resources_.button = LoadTexture("ui", "button", manifest);
  resources_.begin_button = LoadTexture("ui", "begin_button", manifest);
  resources_.settings_menu_background =
      LoadTexture("ui", "settings_menu_background", manifest);
  resources_.slider = LoadTexture("ui", "slider", manifest);
  resources_.checkbox = LoadTexture("ui", "checkbox", manifest);
  resources_.checkmark = LoadTexture("ui", "checkmark", manifest);
  resources_.timer_hourglass = LoadTexture("ui", "hourglass", manifest);

  // Spells stored in `resources_.projectiles` need to be loaded in 2 steps to
  // correctly map the texture file path to the corresponding spell ID.
  std::map<std::string, SDL_Texture*> spell_textures;
  if (manifest["spells"]) {
    for (auto entry : manifest["spells"]) {
      SDL_Texture* tex =
          LoadTexture("spells", entry.first.as<std::string>(), manifest);
      spell_textures[entry.first.as<std::string>()] = tex;
    }
  }

  for (const auto& [spell_id, texture_id] : spell_texture_mapping) {
    SDL_Texture* tex = spell_textures[texture_id];
    if (!tex) {
      std::cerr << "Spell texture not found: " << texture_id << '\n';
      return false;
    }
    resources_.projectiles.push_back(tex);
  }

  return true;
}

bool RenderManager::LoadFonts(const YAML::Node& manifest) {
  if (!manifest["fonts"] || !manifest["fonts"]["november"]) {
    std::cerr << "Missing font in manifest: fonts.november\n";
    return false;
  }

  std::string font_path = manifest["fonts"]["november"].as<std::string>();
  resources_.font_small =
      TTF_OpenFont(font_path.c_str(), ui_config_.fonts.font_size_small);
  resources_.font_medium =
      TTF_OpenFont(font_path.c_str(), ui_config_.fonts.font_size_medium);
  resources_.font_large =
      TTF_OpenFont(font_path.c_str(), ui_config_.fonts.font_size_large);
  resources_.font_huge =
      TTF_OpenFont(font_path.c_str(), ui_config_.fonts.font_size_huge);

  if (!resources_.font_small || !resources_.font_medium ||
      !resources_.font_large || !resources_.font_huge) {
    std::cerr << "TTF font could not be loaded: " << TTF_GetError() << '\n';
    return false;
  }

  return true;
}

bool RenderManager::ValidateTextures() {
  if (!resources_.tile || !resources_.player || !resources_.enemy ||
      !resources_.health_bar || !resources_.level_indicator ||
      !resources_.exp_bar || !resources_.timer_hourglass ||
      !resources_.game_over || !resources_.start_screen ||
      !resources_.level_up_option_card || !resources_.button ||
      !resources_.begin_button || !resources_.settings_menu_background ||
      !resources_.slider || !resources_.checkbox || !resources_.checkmark ||
      !resources_.chest || !resources_.digit_font ||
      std::any_of(resources_.gems.begin(), resources_.gems.end(),
                  [](SDL_Texture* t) { return !t; }) ||
      std::any_of(resources_.items.begin(), resources_.items.end(),
                  [](SDL_Texture* t) { return !t; })) {
    std::cerr << "One or more critical textures failed to load: "
              << SDL_GetError() << '\n';
    return false;
  }
  return true;
}

void RenderManager::LoadUIConfig() {
  ui_config_ = MakeDefaultUIConfig();

  config_manager_.LoadConfigSectionOrDefault(
      "ui.colors", "assets/config/ui/colors.yaml", ui_config_.colors);
  config_manager_.LoadConfigSectionOrDefault(
      "ui.fonts", "assets/config/ui/fonts.yaml", ui_config_.fonts);
  config_manager_.LoadConfigSectionOrDefault(
      "ui.hud", "assets/config/ui/hud.yaml", ui_config_.hud);
  config_manager_.LoadConfigSectionOrDefault(
      "ui.menus", "assets/config/ui/menus.yaml", ui_config_.menus);
  config_manager_.LoadConfigSectionOrDefault(
      "ui.cards", "assets/config/ui/cards.yaml", ui_config_.cards);
  config_manager_.LoadConfigSectionOrDefault(
      "ui.inventory", "assets/config/ui/inventory.yaml", ui_config_.inventory);
}

void RenderManager::Shutdown() {
  // Destroy game textures
  SDL_DestroyTexture(resources_.tile);
  resources_.tile = nullptr;
  SDL_DestroyTexture(resources_.player);
  resources_.player = nullptr;
  SDL_DestroyTexture(resources_.enemy);
  resources_.enemy = nullptr;
  SDL_DestroyTexture(resources_.chest);
  resources_.chest = nullptr;

  for (auto* tex : resources_.projectiles) {
    SDL_DestroyTexture(tex);
  }
  resources_.projectiles.clear();

  for (auto* tex : resources_.gems) {
    SDL_DestroyTexture(tex);
  }
  resources_.gems.clear();

  for (auto* tex : resources_.items) {
    SDL_DestroyTexture(tex);
  }
  resources_.items.clear();

  // Destroy UI textures
  SDL_DestroyTexture(resources_.digit_font);
  resources_.digit_font = nullptr;
  SDL_DestroyTexture(resources_.health_bar);
  resources_.health_bar = nullptr;
  SDL_DestroyTexture(resources_.exp_bar);
  resources_.exp_bar = nullptr;
  SDL_DestroyTexture(resources_.level_indicator);
  resources_.level_indicator = nullptr;
  SDL_DestroyTexture(resources_.timer_hourglass);
  resources_.timer_hourglass = nullptr;
  SDL_DestroyTexture(resources_.game_over);
  resources_.game_over = nullptr;
  SDL_DestroyTexture(resources_.start_screen);
  resources_.start_screen = nullptr;
  SDL_DestroyTexture(resources_.level_up_option_card);
  resources_.level_up_option_card = nullptr;
  SDL_DestroyTexture(resources_.button);
  resources_.button = nullptr;
  SDL_DestroyTexture(resources_.begin_button);
  resources_.begin_button = nullptr;
  SDL_DestroyTexture(resources_.settings_menu_background);
  resources_.settings_menu_background = nullptr;
  SDL_DestroyTexture(resources_.slider);
  resources_.slider = nullptr;
  SDL_DestroyTexture(resources_.checkbox);
  resources_.checkbox = nullptr;
  SDL_DestroyTexture(resources_.checkmark);
  resources_.checkmark = nullptr;

  // Destroy fonts
  if (resources_.font_small) {
    TTF_CloseFont(resources_.font_small);
    resources_.font_small = nullptr;
  }
  if (resources_.font_medium) {
    TTF_CloseFont(resources_.font_medium);
    resources_.font_medium = nullptr;
  }
  if (resources_.font_large) {
    TTF_CloseFont(resources_.font_large);
    resources_.font_large = nullptr;
  }
  if (resources_.font_huge) {
    TTF_CloseFont(resources_.font_huge);
    resources_.font_huge = nullptr;
  }

  IMG_Quit();
  TTF_Quit();

  if (renderer_) {
    SDL_DestroyRenderer(renderer_);
    renderer_ = nullptr;
  }
  if (window_) {
    SDL_DestroyWindow(window_);
    window_ = nullptr;
  }

  SDL_Quit();
}

void RenderManager::SetRenderColor(SDL_Renderer* renderer,
                                   const SDL_Color& color) {
  SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
}

void RenderManager::Render(const Scene& scene, float alpha,
                           const GameStatus& game_status, float time,
                           GameState game_state) {

  SetRenderColor(renderer_, kColorBlack);
  SDL_RenderClear(renderer_);

  if (game_state == GameState::in_start_screen) {
    ui_manager_.UpdateStartScreen();
    UIWidget* start_screen = ui_manager_.GetStartScreenRoot();
    if (start_screen) {
      start_screen->SetVisible(true);
      RenderUITree(start_screen);
      start_screen->SetVisible(false);
    }
  } else {

    Vector2D player_centroid =
        GetCentroid(scene.player.position_, scene.player.stats_.size.GetSize());
    camera_.UpdatePosition(player_centroid);

    camera_.render_position_ =
        Lerp(camera_.prev_position_, camera_.position_, alpha);
    RenderTiledMap();
    RenderPlayer(scene.player, alpha);

    int num_enemy_vertices = SetupEnemyGeometry(scene.enemy, alpha);
    RenderEnemies(num_enemy_vertices);
    SetupProjectileGeometry(scene.projectiles, alpha);
    RenderProjectiles();
    SetupGemGeometry(scene.exp_gem, alpha);
    RenderGem();
    SetupChestGeometry(scene.chest, alpha);
    RenderChests();

    if (game_status.show_occupancy_map) {
      RenderDebugWorldOccupancyMap(scene.occupancy_map);
    }

    if (game_status.show_ray_caster) {
      RenderDebugRayCaster(scene.enemy, alpha);
    }

    RenderUI(time);
    if (game_state == GameState::is_gameover) {
      UIWidget* game_over_screen = ui_manager_.GetGameOverScreenRoot();
      if (game_over_screen) {
        game_over_screen->SetVisible(true);
        RenderUITree(game_over_screen);
        game_over_screen->SetVisible(false);
      }
    } else if (game_state == GameState::in_settings_menu) {
      RenderSettingsMenuState();
    } else if (game_state == GameState::in_level_up) {
      RenderLevelUp();
    } else if (game_state == GameState::in_item_selection) {
      RenderItemSelection();
    } else if (game_state == GameState::in_quit_confirm) {
      RenderQuitConfirmMenu();
    } else if (game_state == GameState::in_chest_opening) {
      UIWidget* chest_screen = ui_manager_.GetChestOpeningRoot();
      if (chest_screen) {
        chest_screen->SetVisible(true);
        RenderUITree(chest_screen);
        chest_screen->SetVisible(false);
      };
    };
  }

  SDL_RenderPresent(renderer_);
};

Vector2D RenderManager::WorldToScreen(Vector2D world_pos) const {
  // Rounding is added to ensure that for textures that use UV coordinates for
  // rendering from a sprite sheet, the sprite does not flicker due to sub-pixel
  // rendering.
  return Round(world_pos - camera_.render_position_);
}

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
      SDL_Rect render_rect = tile_manager_.tiles_[i][j];
      render_rect.x -= static_cast<int>(camera_.render_position_.x);
      render_rect.y -= static_cast<int>(camera_.render_position_.y);
      int tile_id = tile_manager_.tile_map_[i][j];
      const SDL_Rect& source_rect = tile_manager_.select_tiles_[tile_id];
      SDL_RenderCopy(renderer_, resources_.tile, &source_rect, &render_rect);
    }
  }
};

void RenderManager::RenderPlayer(const Player& player, float alpha) {
  Vector2D render_pos = Lerp(player.prev_position_, player.position_, alpha);
  Vector2D screen_pos = WorldToScreen(render_pos);
  float x = screen_pos.x;
  float y = screen_pos.y;

  float w = static_cast<float>(player.stats_.size.GetWidth());
  float h = static_cast<float>(player.stats_.size.GetHeight());

  bool is_standing_still = player.velocity_.Norm() < 1e-3;
  bool is_facing_right = player.last_horizontal_velocity_ >= 0.0f;

  int src_x =
      ((static_cast<int>(SDL_GetTicks64()) / kPlayerAnimationFrameDuration) %
       kPlayerNumSpriteCells) *
      kPlayerSpriteCellWidth;
  int src_y = is_standing_still ? 0 : kPlayerSpriteCellHeight;

  int texture_w, texture_h;
  SDL_QueryTexture(resources_.player, nullptr, nullptr, &texture_w, &texture_h);

  float u_left = static_cast<float>(src_x) / static_cast<float>(texture_w);
  float u_right = static_cast<float>(src_x + kPlayerSpriteCellWidth) /
                  static_cast<float>(texture_w);
  float v_top = static_cast<float>(src_y) / static_cast<float>(texture_h);
  float v_bottom = static_cast<float>(src_y + kPlayerSpriteCellHeight) /
                   static_cast<float>(texture_h);

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

  SDL_RenderGeometry(renderer_, resources_.player, vertices, 6, nullptr, 0);
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

    float w = static_cast<float>(enemy.sprite_size[i].width);
    float h = static_cast<float>(enemy.sprite_size[i].height);

    // Skip setting up the enemy geometry if they are not in view.
    if (enemy.position[i].x + w < cull_left ||
        enemy.position[i].x > cull_right ||
        enemy.position[i].y + h < cull_top ||
        enemy.position[i].y > cull_bottom) {
      continue;
    }

    Vector2D render_pos =
        Lerp(enemy.prev_position[i], enemy.position[i], alpha);
    Vector2D screen_pos = WorldToScreen(render_pos);
    float x = screen_pos.x;
    float y = screen_pos.y;

    uint16_t time_offset = i * 127;
    uint16_t frame_idx =
        ((SDL_GetTicks64() + time_offset) / kEnemyAnimationFrameDuration) %
        kEnemyNumSpriteCells;

    float u_left = static_cast<float>(frame_idx) * cell_uv_width;
    float u_right = u_left + cell_uv_width;
    float v_top = kTexCoordTop;
    float v_bottom = kTexCoordBottom;

    bool is_facing_right = enemy.last_horizontal_velocity[i] > 0;

    float vertex_left = is_facing_right ? u_left : u_right;
    float vertex_right = is_facing_right ? u_right : u_left;

    // Vertices for triangle 1 (top-left, bottom-left, bottom-right)
    // top-left
    enemies_vertices_[current_vertex_idx + 0] = {
        {x, y}, {255, 255, 255, 255}, {vertex_left, v_top}};
    // bottom-left
    enemies_vertices_[current_vertex_idx + 1] = {
        {x, y + h}, {255, 255, 255, 255}, {vertex_left, v_bottom}};
    // bottom-right
    enemies_vertices_[current_vertex_idx + 2] = {
        {x + w, y + h}, {255, 255, 255, 255}, {vertex_right, v_bottom}};
    // Vertices for triangle 2 (top-left, bottom-right, top-right)
    // top-left (copy)
    enemies_vertices_[current_vertex_idx + 3] =
        enemies_vertices_[current_vertex_idx + 0];
    // bottom-right (copy)
    enemies_vertices_[current_vertex_idx + 4] =
        enemies_vertices_[current_vertex_idx + 2];
    // top-right
    enemies_vertices_[current_vertex_idx + 5] = {
        {x + w, y}, {255, 255, 255, 255}, {vertex_right, v_top}};

    current_vertex_idx += kEnemyVertices;
  }
  return current_vertex_idx;
};

void RenderManager::RenderEnemies(int num_vertices) {
  // We use the number of vertices calculated during the setup of the enemy
  // geometry to render the vertices.
  SDL_RenderGeometry(renderer_, resources_.enemy, enemies_vertices_,
                     num_vertices, nullptr, 0);
};

void RenderManager::SetupProjectileGeometry(const Projectiles& projectiles,
                                            float alpha) {
  projectile_vertices_grouped_.clear();
  size_t num_projectiles = projectiles.GetNumProjectiles();
  if (num_projectiles == 0) {
    return;
  }

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
    float w = static_cast<float>(projectiles.sprite_size_[i].width);
    float h = static_cast<float>(projectiles.sprite_size_[i].height);

    // Skip setting up the projectile geometry if they are not in view.
    if (projectiles.position_[i].x + w < cull_left ||
        projectiles.position_[i].x > cull_right ||
        projectiles.position_[i].y + h < cull_top ||
        projectiles.position_[i].y > cull_bottom) {
      continue;
    }

    Vector2D render_pos =
        Lerp(projectiles.prev_position_[i], projectiles.position_[i], alpha);
    Vector2D screen_pos = WorldToScreen(render_pos);
    float x = screen_pos.x;
    float y = screen_pos.y;

    int texture_id = projectiles.proj_type_[i];

    int time_offset = i * 127;
    int frame_idx = ((static_cast<int>(SDL_GetTicks64()) + time_offset) /
                     kProjectileAnimationFrameDuration) %
                    kProjectileNumSpriteCells;

    float u_left = static_cast<float>(frame_idx) * cell_uv_width;
    float u_right = u_left + cell_uv_width;
    float v_top = kTexCoordTop;
    float v_bottom = kTexCoordBottom;

    bool is_facing_right = projectiles.direction_[i].x > 0;

    float vertex_left = is_facing_right ? u_left : u_right;
    float vertex_right = is_facing_right ? u_right : u_left;

    SDL_Vertex vertices[kProjectileVertices];

    // Vertices for triangle 1 (top-left, bottom-left, bottom-right)
    // top-left
    vertices[0] = {{x, y}, {255, 255, 255, 255}, {vertex_left, v_top}};
    // bottom-left
    vertices[1] = {{x, y + h}, {255, 255, 255, 255}, {vertex_left, v_bottom}};
    // bottom_right
    vertices[2] = {
        {x + w, y + h}, {255, 255, 255, 255}, {vertex_right, v_bottom}};
    // Vertices for triangle 2 (top-left, bottom-right, top-right)
    // top-left (copy)
    vertices[3] = vertices[0];
    // bottom-right (copy)
    vertices[4] = vertices[2];
    // top-right
    vertices[5] = {{x + w, y}, {255, 255, 255, 255}, {vertex_right, v_top}};

    for (int j = 0; j < kProjectileVertices; ++j) {
      projectile_vertices_grouped_[texture_id].push_back(vertices[j]);
    }
  }
};

void RenderManager::RenderProjectiles() {
  for (const auto& pair : projectile_vertices_grouped_) {
    int texture_id = pair.first;
    const std::vector<SDL_Vertex>& vertices = pair.second;
    if (texture_id >= 0 && texture_id < resources_.projectiles.size()) {
      SDL_RenderGeometry(renderer_, resources_.projectiles[texture_id],
                         vertices.data(), static_cast<int>(vertices.size()),
                         nullptr, 0);
    };
  };
};

void RenderManager::SetupGemGeometry(const ExpGem& exp_gem, float alpha) {
  gem_vertices_grouped_.clear();
  size_t num_gems = exp_gem.GetNumExpGems();
  if (num_gems == 0) {
    return;
  }

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
    float w = static_cast<float>(exp_gem.sprite_size_[i].width);
    float h = static_cast<float>(exp_gem.sprite_size_[i].height);

    // Skip setting up the exp gem geometry if they are not in view.
    if (exp_gem.position_[i].x + w < cull_left ||
        exp_gem.position_[i].x > cull_right ||
        exp_gem.position_[i].y + h < cull_top ||
        exp_gem.position_[i].y > cull_bottom) {
      continue;
    }

    Vector2D render_pos =
        Lerp(exp_gem.prev_position_[i], exp_gem.position_[i], alpha);
    Vector2D screen_pos = WorldToScreen(render_pos);
    float x = screen_pos.x;
    float y = screen_pos.y;

    int texture_id = static_cast<int>(exp_gem.rarity_[i]);

    float u_left = 0.0f;
    float u_right = cell_uv_width;
    float v_top = kTexCoordTop;
    float v_bottom = kTexCoordBottom;

    float vertex_left = u_left;
    float vertex_right = u_right;

    SDL_Vertex vertices[kExpGemVertices];

    // Vertices for triangle 1 (top-left, bottom-left, bottom-right)
    // top-left
    vertices[0] = {{x, y}, {255, 255, 255, 255}, {vertex_left, v_top}};
    // bottom-left
    vertices[1] = {{x, y + h}, {255, 255, 255, 255}, {vertex_left, v_bottom}};
    // bottom_right
    vertices[2] = {
        {x + w, y + h}, {255, 255, 255, 255}, {vertex_right, v_bottom}};
    // Vertices for triangle 2 (top-left, bottom-right, top-right)
    // top-left (copy)
    vertices[3] = vertices[0];
    // bottom-right (copy)
    vertices[4] = vertices[2];
    // top-right
    vertices[5] = {{x + w, y}, {255, 255, 255, 255}, {vertex_right, v_top}};

    for (int j = 0; j < kExpGemVertices; ++j) {
      gem_vertices_grouped_[texture_id].push_back(vertices[j]);
    }
  }
};

void RenderManager::RenderGem() {
  for (const auto& pair : gem_vertices_grouped_) {
    int texture_id = pair.first;
    const std::vector<SDL_Vertex>& vertices = pair.second;
    if (texture_id >= 0 && texture_id < resources_.gems.size()) {
      SDL_RenderGeometry(renderer_, resources_.gems[texture_id],
                         vertices.data(), static_cast<int>(vertices.size()),
                         nullptr, 0);
    };
  };
};

void RenderManager::SetupChestGeometry(const Chest& chest, float alpha) {
  chest_vertices_.clear();
  size_t num_chests = chest.GetNumChests();
  if (num_chests == 0) {
    return;
  }
  int texture_w, texture_h;
  SDL_QueryTexture(resources_.chest, nullptr, nullptr, &texture_w, &texture_h);
  float cell_uv_width = 1.0f / static_cast<float>(kChestSpriteSheetCols);
  float cell_uv_height = 1.0f / static_cast<float>(kChestSpriteSheetRows);
  // Frame 0 = top-left cell (closed chest)
  float u_left = 0.0f;
  float u_right = cell_uv_width;
  float v_top = 0.0f;
  float v_bottom = cell_uv_height;
  float cull_left = camera_.render_position_.x - kRenderCullPadding;
  float cull_right =
      camera_.render_position_.x + kWindowWidth + kRenderCullPadding;
  float cull_top = camera_.render_position_.y - kRenderCullPadding;
  float cull_bottom =
      camera_.render_position_.y + kWindowHeight + kRenderCullPadding;
  for (size_t i = 0; i < num_chests; ++i) {
    float w = static_cast<float>(chest.sprite_size_[i].width);
    float h = static_cast<float>(chest.sprite_size_[i].height);
    if (chest.position_[i].x + w < cull_left ||
        chest.position_[i].x > cull_right ||
        chest.position_[i].y + h < cull_top ||
        chest.position_[i].y > cull_bottom) {
      continue;
    }

    Vector2D render_pos =
        Lerp(chest.prev_position_[i], chest.position_[i], alpha);
    Vector2D screen_pos = WorldToScreen(render_pos);
    float x = screen_pos.x;
    float y = screen_pos.y;

    SDL_Color c = {255, 255, 255, 255};
    SDL_Vertex vertices[kChestVertices] = {
        {{x, y}, c, {u_left, v_top}},
        {{x, y + h}, c, {u_left, v_bottom}},
        {{x + w, y + h}, c, {u_right, v_bottom}},
        {{x, y}, c, {u_left, v_top}},
        {{x + w, y + h}, c, {u_right, v_bottom}},
        {{x + w, y}, c, {u_right, v_top}}};
    for (int j = 0; j < kChestVertices; ++j) {
      chest_vertices_.push_back(vertices[j]);
    }
  }
}

void RenderManager::RenderChests() {
  if (chest_vertices_.empty()) {
    return;
  }
  SDL_RenderGeometry(renderer_, resources_.chest, chest_vertices_.data(),
                     static_cast<int>(chest_vertices_.size()), nullptr, 0);
}

void RenderManager::RenderDebugWorldOccupancyMap(
    const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map) {
  // Get the original blend mode to be able to later restore it. The debug
  // visualization should blend textures, but regular rendering should not.
  SDL_BlendMode original_blend_mode;
  SDL_GetRenderDrawBlendMode(renderer_, &original_blend_mode);
  SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);

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
      render_rect.x =
          static_cast<int>(static_cast<float>(i * kOccupancyMapResolution) -
                           camera_.render_position_.x);
      render_rect.y =
          static_cast<int>(static_cast<float>(j * kOccupancyMapResolution) -
                           camera_.render_position_.y);
      render_rect.w = kOccupancyMapResolution;
      render_rect.h = kOccupancyMapResolution;

      uint16_t mask = occupancy_map.GetMask(i, j);

      if (mask != kMaskTypeNone) {
        int r = 0, g = 0, b = 0, a = 0;
        int count = 0;

        if (mask & kMaskTypePlayer) {
          r += kColorBlue.r;
          g += kColorBlue.g;
          b += kColorBlue.b;
          a += 128;
          count++;
        }
        if (mask & kMaskTypeEnemy) {
          r += kColorRed.r;
          g += kColorRed.g;
          b += kColorRed.b;
          a += 128;
          count++;
        }
        if (mask & kMaskTypeProjectile) {
          r += kColorYellow.r;
          g += kColorYellow.g;
          b += kColorYellow.b;
          a += 128;
          count++;
        }
        if (mask & kMaskTypeTerrain) {
          r += kColorGreen.r;
          g += kColorGreen.g;
          b += kColorGreen.b;
          a += 128;
          count++;
        }

        if (count > 0) {
          SDL_SetRenderDrawColor(renderer_, r / count, g / count, b / count,
                                 a / count);
        } else {
          // Fallback for types not handled explicitly above
          SetRenderColor(renderer_, WithOpacity(kColorGrey, 128));
        }
        // The rectangles are rendered first so the grid cells are on top.
        SDL_RenderFillRect(renderer_, &render_rect);
      }

      SetRenderColor(renderer_, WithOpacity(kColorBlack, 50));
      SDL_RenderDrawRect(renderer_, &render_rect);
    }
  }

  SDL_SetRenderDrawBlendMode(renderer_, original_blend_mode);
};

void RenderManager::RenderDebugRayCaster(const Enemy& enemy, float alpha) {
  SDL_BlendMode original_blend_mode;
  SDL_GetRenderDrawBlendMode(renderer_, &original_blend_mode);
  SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);

  for (int i = 0; i < kNumEnemies; ++i) {
    if (!enemy.is_alive[i]) {
      continue;
    }

    Vector2D enemy_pos_world =
        Lerp(enemy.prev_position[i], enemy.position[i], alpha);

    Vector2D center_world = enemy_pos_world + enemy.collider[i].offset;

    float half_w = static_cast<float>(enemy.collider[i].size.width) * 0.5f;
    float half_h = static_cast<float>(enemy.collider[i].size.height) * 0.5f;
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
          SetRenderColor(renderer_, WithOpacity(kColorRed, 150));
          break;
        case EntityType::terrain:
          SetRenderColor(renderer_, WithOpacity(kColorGrey, 50));
          break;
        case EntityType::enemy:
          SetRenderColor(renderer_, WithOpacity(kColorOrange, 150));
          break;
        default:
          SetRenderColor(renderer_, WithOpacity(kColorBlack, 50));
          break;
      }

      SDL_RenderDrawLine(renderer_, static_cast<int>(start_screen.x),
                         static_cast<int>(start_screen.y),
                         static_cast<int>(end_screen.x),
                         static_cast<int>(end_screen.y));
    }
  }

  SDL_SetRenderDrawBlendMode(renderer_, original_blend_mode);
}
void RenderManager::RenderUI(float time) {
  ui_manager_.UpdateTimer(time);
  RenderUITree(ui_manager_.GetRootWidget());
}

void RenderManager::RenderUITree(UIWidget* root) {
  if (!root || !root->IsVisible()) {
    return;
  }
  RenderWidgetRecursive(root);
}

void RenderManager::RenderWidgetRecursive(UIWidget* widget) {
  if (!widget || !widget->IsVisible()) {
    return;
  }

  SDL_Rect bounds = widget->GetComputedBounds();

  switch (widget->GetWidgetType()) {
    case WidgetType::Panel: {
      auto* panel = static_cast<Panel*>(widget);
      if (panel->HasBackgroundColor()) {
        SDL_BlendMode original_blend_mode;
        SDL_GetRenderDrawBlendMode(renderer_, &original_blend_mode);
        SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);

        SDL_Color color = panel->GetBackgroundColor();
        SDL_SetRenderDrawColor(renderer_, color.r, color.g, color.b, color.a);
        SDL_RenderFillRect(renderer_, &bounds);

        SDL_SetRenderDrawBlendMode(renderer_, original_blend_mode);
      }

      if (panel->GetBackgroundTexture()) {
        SDL_Rect src = panel->GetBackgroundSrcRect();
        SDL_Rect* src_ptr = (src.w > 0 && src.h > 0) ? &src : nullptr;
        SDL_RenderCopy(renderer_, panel->GetBackgroundTexture(), src_ptr,
                       &bounds);
      }
      break;
    }

    case WidgetType::Image: {
      auto* img = static_cast<UIImage*>(widget);
      if (img->GetTexture()) {
        SDL_Rect src = img->GetSrcRect();
        SDL_Rect* src_ptr = (src.w > 0 && src.h > 0) ? &src : nullptr;
        SDL_RenderCopy(renderer_, img->GetTexture(), src_ptr, &bounds);
      }
      break;
    }

    case WidgetType::Animation: {
      auto* anim_img = static_cast<UIAnimation*>(widget);
      if (anim_img->GetTexture()) {
        SDL_Rect src = anim_img->GetCurrentSrcRect();
        SDL_RenderCopy(renderer_, anim_img->GetTexture(), &src, &bounds);
      }
      break;
    }

    case WidgetType::Label: {
      auto* lbl = static_cast<UILabel*>(widget);
      if (lbl->GetUseDigitFont()) {
        Size2D sprite_size = {
            static_cast<uint32_t>(lbl->GetDigitSpriteWidth()),
            static_cast<uint32_t>(lbl->GetDigitSpriteHeight())};
        Size2D char_size = {static_cast<uint32_t>(lbl->GetCharWidth()),
                            static_cast<uint32_t>(lbl->GetCharHeight())};
        RenderDigitString(lbl->GetText(), bounds.x, bounds.y, sprite_size,
                          char_size);
      } else if (lbl->GetFont()) {
        RenderText(lbl->GetText(), {bounds.x, bounds.y}, lbl->GetColor(),
                   lbl->GetFont(),
                   {lbl->GetCenterWidth(), lbl->GetWrapWidth()});
      }
      break;
    }

    case WidgetType::Checkbox: {
      auto* chk = static_cast<UICheckbox*>(widget);
      if (chk->GetBoxTexture()) {
        SDL_Rect src = chk->GetCurrentBoxSrcRect();
        SDL_RenderCopy(renderer_, chk->GetBoxTexture(), &src, &bounds);
      }
      if (chk->IsChecked() && chk->GetMarkTexture()) {
        SDL_Rect src = chk->GetMarkSrcRect();
        SDL_RenderCopy(renderer_, chk->GetMarkTexture(), &src, &bounds);
      }
      break;
    }

    case WidgetType::Button: {
      auto* btn = static_cast<UIButton*>(widget);
      if (btn->GetTexture()) {
        SDL_Rect src = btn->GetCurrentSrcRect();
        SDL_RenderCopy(renderer_, btn->GetTexture(), &src, &bounds);
      }
      if (btn->GetLabelFont() && !btn->GetLabel().empty()) {
        RenderText(btn->GetLabel(), {bounds.x, bounds.y + (bounds.h - 26) / 2},
                   {255, 255, 255, 255}, btn->GetLabelFont(),
                   {static_cast<float>(bounds.w), 0.0f});
      }
      break;
    }

    case WidgetType::ProgressBar: {
      auto* bar = static_cast<UIProgressBar*>(widget);
      if (bar->GetContainerTexture()) {
        SDL_Rect src = bar->GetContainerSrcRect();
        SDL_RenderCopy(renderer_, bar->GetContainerTexture(), &src, &bounds);
      }
      if (bar->GetFillTexture()) {
        SDL_Rect fill_src = bar->GetClippedFillSrcRect();
        SDL_Rect fill_dst = bar->GetFillDestRect();
        SDL_RenderCopy(renderer_, bar->GetFillTexture(), &fill_src, &fill_dst);
      }
      break;
    }

    case WidgetType::InventoryItem: {
      auto* inv_item = static_cast<UIInventoryItem*>(widget);
      if (inv_item->GetItemTexture()) {
        SDL_Rect dest_rect = bounds;
        int icon_size =
            static_cast<int>(ui_config_.inventory.inventory_icon_size);
        dest_rect.y += (bounds.h - icon_size) / 2;
        dest_rect.w = icon_size;
        dest_rect.h = icon_size;
        SDL_RenderCopy(renderer_, inv_item->GetItemTexture(), nullptr,
                       &dest_rect);
      }
      break;
    }

    default:
      break;
  }

  for (auto& child : widget->GetChildren()) {
    RenderWidgetRecursive(child.get());
  }
}

void RenderManager::RenderDigitString(const std::string& text, int start_x,
                                      int start_y, Size2D sprite_size,
                                      Size2D char_size) {

  int char_width = static_cast<int>(char_size.width);
  int char_height = static_cast<int>(char_size.height);

  int current_x = start_x;

  for (char c : text) {
    SDL_Rect src_rect = {0, 0, static_cast<int>(sprite_size.width),
                         static_cast<int>(sprite_size.height)};

    if (c >= '0' && c <= '9') {
      int digit = c - '0';
      src_rect.x = static_cast<int>(digit * sprite_size.width);
    } else if (c == '/') {
      // TODO: Add more maintainable way of getting the sprite cell for a char.
      src_rect.x = static_cast<int>(sprite_size.width * 10);
    } else if (c == '-') {
      src_rect.x = static_cast<int>(sprite_size.width * 11);
    } else {
      // if not one of the above, they are not in the texture atlas, so we skip.
      current_x += char_width;
      continue;
    }

    SDL_Rect dest_rect = {current_x, start_y, char_width, char_height};
    SDL_RenderCopy(renderer_, resources_.digit_font, &src_rect, &dest_rect);

    current_x += char_width;
  };
};

void RenderManager::RenderSettingsMenuState() {
  UIWidget* settings = ui_manager_.GetSettingsRoot();

  if (settings) {
    settings->SetVisible(true);
    RenderUITree(settings);
    settings->SetVisible(false);
  }
}

void RenderManager::UpdateSettingsMenuState(float volume, bool is_muted,
                                            const GameStatus& game_status) {
  ui_manager_.UpdateSettingsMenu(volume, is_muted, game_status);
}

void RenderManager::RenderLevelUp() {
  ui_manager_.UpdateLevelUpMenu();

  UIWidget* level_up = ui_manager_.GetLevelUpRoot();
  if (level_up) {
    level_up->SetVisible(true);
    RenderWidgetRecursive(level_up);
    level_up->SetVisible(false);
  }
}

void RenderManager::RenderItemSelection() {
  ui_manager_.UpdateItemMenu();

  UIWidget* item_menu = ui_manager_.GetItemMenuRoot();
  if (item_menu) {
    item_menu->SetVisible(true);
    RenderWidgetRecursive(item_menu);
    item_menu->SetVisible(false);
  }
}

void RenderManager::RenderQuitConfirmMenu() {
  ui_manager_.UpdateQuitConfirmMenu();

  UIWidget* quit_confirm = ui_manager_.GetQuitConfirmRoot();
  if (quit_confirm) {
    quit_confirm->SetVisible(true);
    RenderUITree(quit_confirm);
    quit_confirm->SetVisible(false);
  }
}

// Render a string of text at a specified location (x,y) with a given color
// and font.
// Optional: If you specify a positive center_width in TextLayout, then it
// will center-align the text along that center_width.
void RenderManager::RenderText(const std::string& text, SDL_Point pos,
                               SDL_Color color, TTF_Font* font,
                               TextLayout layout) {
  if (layout.wrap_width > 0.0f) {
    TTF_SetFontWrappedAlign(font, TTF_WRAPPED_ALIGN_CENTER);
  }
  SDL_Surface* surface =
      layout.wrap_width > 0.0f
          ? TTF_RenderText_Blended_Wrapped(
                font, text.c_str(), color,
                static_cast<Uint32>(layout.wrap_width))
          : TTF_RenderText_Blended(font, text.c_str(), color);
  if (layout.wrap_width > 0.0f) {
    TTF_SetFontWrappedAlign(font, TTF_WRAPPED_ALIGN_LEFT);
  }
  SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer_, surface);

  int render_x = pos.x;
  if (layout.center_width > 0.0f) {
    render_x = pos.x + static_cast<int>((layout.center_width -
                                         static_cast<float>(surface->w)) /
                                        2.0f);
  }

  SDL_Rect dest = {render_x, pos.y, surface->w, surface->h};
  SDL_RenderCopy(renderer_, texture, nullptr, &dest);

  SDL_DestroyTexture(texture);
  SDL_FreeSurface(surface);
}

}  // namespace arelto
