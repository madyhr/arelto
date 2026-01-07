// include/render_manager.h
#ifndef RL2_RENDER_MANAGER_H_
#define RL2_RENDER_MANAGER_H_
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_ttf.h>
#include <map>
#include <string>
#include "constants/enemy.h"
#include "constants/game.h"
#include "constants/map.h"
#include "entity.h"
#include "map.h"
#include "scene.h"
#include "types.h"
#include "ui_manager.h"

namespace rl2 {

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
  SDL_Rect map_layout = {0, 0, kWindowWidth, kWindowHeight};
  TileManager tile_manager;
  UIResources ui_resources;
};

class RenderManager {

 public:
  RenderManager();
  ~RenderManager();

  bool Initialize(bool is_headless);
  void Shutdown();

  void Render(const Scene& scene, float alpha, bool debug_mode, float time,
              GameState game_state);
  void RenderGameOver();
  void RenderStartScreen();
  void RenderPaused();

  Camera camera_;

 private:
  RenderResources resources_;
  UIManager ui_manager_;

  bool InitializeCamera(const Player& player);
  void UpdateCameraPosition(const Player& player);
  void RenderTiledMap();
  void RenderPlayer(const Player& player, float alpha);
  int SetupEnemyGeometry(const Enemy& enemy, float alpha);
  void RenderEnemies(const Enemy& enemy, int num_vertices);
  void SetupProjectileGeometry(const Projectiles& projectiles, float alpha);
  void RenderProjectiles(const Projectiles& projectiles);
  void RenderDebugWorldOccupancyMap(
      const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map);
  void RenderDebugEnemyOccupancyMap(
      const Enemy& enemy,
      const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map,
      float alpha);
  void RenderDebugRayCaster(const Enemy& enemy, float alpha);
  void RenderUI(const Scene& scene, float time);
  void RenderDigitString(const std::string& text, int start_x, int start_y,
                         Size2D sprite_size, Size2D char_size);
};

}  // namespace rl2
#endif
