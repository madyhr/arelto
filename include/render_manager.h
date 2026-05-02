// include/render_manager.h
#ifndef RL2_RENDER_MANAGER_H_
#define RL2_RENDER_MANAGER_H_
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_ttf.h>
#include <map>
#include <string>
#include "config/config_manager.h"
#include "config/ui_config.h"
#include "constants/enemy.h"
#include "constants/game.h"
#include "constants/map.h"
#include "entity.h"
#include "spell_manager.h"
#include "event_manager.h"
#include "map.h"
#include "scene.h"
#include "types.h"
#include "ui/widget.h"
#include "ui_manager.h"

namespace arelto {

struct TextLayout {
  int center_width;
  int wrap_width;
};

struct RenderResources {
  SDL_Window* window = nullptr;
  SDL_Renderer* renderer = nullptr;
  SDL_Texture* map_texture = nullptr;
  SDL_Texture* tile_texture = nullptr;
  SDL_Texture* player_texture = nullptr;
  SDL_Texture* enemy_texture = nullptr;
  SDL_Vertex enemies_vertices_[kTotalEnemyVertices];
  std::vector<SDL_Texture*> projectile_textures;
  std::map<int, std::vector<SDL_Vertex>> projectile_vertices_grouped_;
  std::vector<SDL_Texture*> gem_textures;
  std::map<int, std::vector<SDL_Vertex>> gem_vertices_grouped_;
  SDL_Texture* chest_texture = nullptr;
  std::vector<SDL_Vertex> chest_vertices_;
  std::vector<SDL_Texture*> item_textures;
  SDL_Rect map_layout = {0, 0, kWindowWidth, kWindowHeight};
  TileManager tile_manager;
  UIResources ui_resources;
};

class RenderManager {

 public:
  RenderManager();
  ~RenderManager();

  bool Initialize(bool is_headless, EventManager& event_manager,
                      const std::vector<std::string>& texture_paths);
  void Shutdown();

  void Render(const Scene& scene, float alpha, const GameStatus& game_status,
              float time, GameState game_state);

  void RenderSettingsMenuState();
  void RenderLevelUp();
  void RenderItemSelection();
  void RenderQuitConfirmMenu();
  void UpdateSettingsMenuState(float volume, bool is_muted,
                               const GameStatus& game_status);

  UIManager& GetUIManager() { return ui_manager_; }

  Camera camera_;

 private:
  RenderResources resources_;
  UIManager ui_manager_;
  ConfigManager config_manager_;
  UIConfig ui_config_ = MakeDefaultUIConfig();

  Vector2D WorldToScreen(Vector2D world_pos) const;
  void LoadUIConfig();
  void SetRenderColor(SDL_Renderer* renderer, const SDL_Color& color);
  bool InitializeCamera(const Player& player);
  void RenderTiledMap();
  void RenderPlayer(const Player& player, float alpha);
  int SetupEnemyGeometry(const Enemy& enemy, float alpha);
  void RenderEnemies(int num_vertices);
  void SetupProjectileGeometry(const Projectiles& projectiles, float alpha);
  void RenderProjectiles();
  void SetupGemGeometry(const ExpGem& exp_gem, float alpha);
  void RenderGem();
  void SetupChestGeometry(const Chest& chest, float alpha);
  void RenderChests();
  void RenderDebugWorldOccupancyMap(
      const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map);
  void RenderDebugRayCaster(const Enemy& enemy, float alpha);

  // Widget tree UI rendering
  void RenderUI(float time);
  void RenderUITree(UIWidget* root);
  void RenderWidgetRecursive(UIWidget* widget);

  // Text primitives
  void RenderDigitString(const std::string& text, int start_x, int start_y,
                         Size2D sprite_size, Size2D char_size);
  void RenderText(const std::string& text, SDL_Point pos, SDL_Color color,
                  TTF_Font* font, TextLayout layout = {0, 0});
};

}  // namespace arelto
#endif
