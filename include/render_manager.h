// include/render_manager.h
#ifndef RL2_RENDER_MANAGER_H_
#define RL2_RENDER_MANAGER_H_
#include <SDL2/SDL_render.h>
#include <map>
#include "constants.h"
#include "entity.h"
#include "map.h"
#include "types.h"

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
};

class RenderManager {

 public:
  RenderManager();
  ~RenderManager();

  bool Initialize(bool is_headless);
  void Shutdown();

  void Render(
      const Player& player, const Enemy& enemy, const Projectiles& projectiles,
      const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map,
      float alpha, bool debug_mode = false);

  Camera camera_;

 private:
  RenderResources resources_;

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
};

}  // namespace rl2
#endif
