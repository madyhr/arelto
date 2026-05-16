// include/render_manager.h
#ifndef RL2_RENDER_MANAGER_H_
#define RL2_RENDER_MANAGER_H_
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_ttf.h>
#include <map>
#include <string>
#include <vector>
#include "abilities.h"
#include "config/config_manager.h"
#include "config/ui_config.h"
#include "constants/enemy.h"
#include "constants/map.h"
#include "entity.h"
#include "event_manager.h"
#include "map.h"
#include "render_resources.h"
#include "scene.h"
#include "types.h"
#include "ui/widget.h"
#include "ui/widgets.h"
#include "ui_manager.h"

namespace arelto {

struct TextLayout {
  float center_width = 0.0f;
  float wrap_width = 0.0f;
  float container_height = 0.0f;
  TextVerticalAlign vertical_align = TextVerticalAlign::top;
};

class RenderManager {

 public:
  RenderManager();
  ~RenderManager();

  bool Initialize(bool is_headless, EventManager& event_manager,
                  const SpellTextureMapping& spell_texture_mapping);
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
  // SDL resources
  SDL_Window* window_ = nullptr;
  SDL_Renderer* renderer_ = nullptr;

  RenderResources resources_;

  TileManager tile_manager_;
  UIManager ui_manager_;
  ConfigManager config_manager_;
  UIConfig ui_config_ = MakeDefaultUIConfig();

  // Per-frame vertex buffers
  SDL_Vertex enemies_vertices_[kTotalEnemyVertices];
  std::map<int, std::vector<SDL_Vertex>> projectile_vertices_grouped_;
  std::map<int, std::vector<SDL_Vertex>> gem_vertices_grouped_;
  std::vector<SDL_Vertex> chest_vertices_;

  Vector2D WorldToScreen(Vector2D world_pos) const;
  void LoadUIConfig();
  bool LoadTextures(const YAML::Node& manifest,
                    const SpellTextureMapping& spell_texture_mapping);
  bool LoadFonts(const YAML::Node& manifest);
  bool ValidateTextures();
  SDL_Texture* LoadTexture(const std::string& section, const std::string& key,
                           const YAML::Node& manifest);
  void SetRenderColor(SDL_Renderer* renderer, const SDL_Color& color);
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
                  TTF_Font* font, TextLayout layout = {0.0f, 0.0f});
};

}  // namespace arelto
#endif
