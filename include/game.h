// include/game.h
#ifndef RL2_GAME_H_
#define RL2_GAME_H_

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL_render.h>
#include <array>
#include <cstdint>
#include <vector>
#include "constants.h"
#include "entity.h"
#include "map.h"

namespace rl2 {
// Struct to hold all the core resources
struct GameResources {
  SDL_Window* window = nullptr;
  SDL_Renderer* renderer = nullptr;
  SDL_Texture* map_texture = nullptr;
  SDL_Texture* tile_texture = nullptr;
  SDL_Texture* player_texture = nullptr;
  SDL_Texture* enemy_texture = nullptr;
  SDL_Texture* projectile_texture = nullptr;
  SDL_Rect map_layout = {(int)0, (int)0, kWindowWidth, kWindowHeight};
  TileManager tile_manager;
};

class Camera {
public:
	Vector2D position_;
};

class FrameStats {
 public:
  std::array<float, kFrameTimes> frame_time_buffer{};
  float frame_time_sum = 0.0f;
  int current_buffer_length = 0;
  int max_buffer_length = 100;
  int head_index = 0;
  void update_frame_time_buffer(float dt);
  float get_average_frame_time();
  void print_fps_running_average(float dt);
};

struct GameStatus {
  FrameStats frame_stats;
};

class Game {
 public:
  Game();
  ~Game();
  bool Initialize();
  void RunGameLoop();
  void Shutdown();

 private:
  GameResources resources_;
  GameStatus game_status_;
  Player player_;
  Enemies enemy_;
  Projectiles projectiles_;
  Camera camera_;
  SDL_Vertex enemies_vertices_[kTotalEnemyVertices];
  std::vector<SDL_Vertex> projectiles_vertices_;
  bool is_running_;
  uint64_t ticks_count_ = 0;
  bool InitializeResources();
  bool InitializePlayer();
  bool InitializeEnemies();
  bool InitializeCamera();
  void ProcessInput();
  void ProcessPlayerInput();
  void Update();
  void GenerateOutput();
  void RenderTiledMap();
  void SetupEnemyGeometry();
  void SetupProjectileGeometry();
  void UpdateEnemyPosition(float dt);
  void UpdatePlayerPosition(float dt);
  void UpdateProjectilePosition(float dt);
  void UpdateCameraPosition();
  void HandleCollisions();
  void HandleOutOfBounds();
};

}  // namespace rl2
#endif
