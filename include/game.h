// include/game.h
#ifndef RL2_GAME_H_
#define RL2_GAME_H_

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_render.h>
#include <array>
#include <map>
#include <vector>
#include "constants.h"
#include "entity.h"
#include "map.h"
#include "types.h"

namespace rl2 {
// Struct to hold all the core resources
struct GameResources {
  SDL_Window* window = nullptr;
  SDL_Renderer* renderer = nullptr;
  SDL_Texture* map_texture = nullptr;
  SDL_Texture* tile_texture = nullptr;
  SDL_Texture* player_texture = nullptr;
  SDL_Texture* enemy_texture = nullptr;
  std::vector<SDL_Texture*> projectile_textures;
  SDL_Rect map_layout = {0, 0, kWindowWidth, kWindowHeight};
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
  bool in_debug_mode;
  bool in_headless_mode;
};

enum GameState : int {
  in_start_screen = 0,
  in_main_menu,
  is_running,
  is_gameover,
  in_shutdown,
};

class Game {
 public:
  Game();
  ~Game();
  bool Initialize();
  void RunGameLoop();
  void Shutdown();
  void FillObservationBuffer(float* buffer_ptr, int buffer_size);
  int GetObservationSize();
  int GetGameState();
  void SetGameState(int game_state);

 private:
  GameResources resources_;
  GameStatus game_status_;
  GameState game_state_;
  Player player_;
  Enemy enemy_;
  Projectiles projectiles_;
  FixedMap<(int)kMapWidth / kOccupancyMapResolution,
           (int)kMapHeight / kOccupancyMapResolution>
      world_occupancy_map_;
  Camera camera_;
  Vector2D cursor_position_;
  SDL_Vertex enemies_vertices_[kTotalEnemyVertices];
  std::map<int, std::vector<SDL_Vertex>> projectile_vertices_grouped_;
  bool in_debug_mode = true;
  float last_kill_tick = 0.0f;
  float time_ = 0.0f;
  float dt = 0.0f;
  bool InitializeResources();
  bool InitializePlayer();
  bool InitializeEnemies();
  bool InitializeCamera();
  void ProcessInput();
  Vector2D GetCursorPositionWorld();
  void ProcessPlayerInput();
  void Update();
  void UpdateEnemyPosition(float dt);
  void UpdatePlayerPosition(float dt);
  void UpdateProjectilePosition(float dt);
  void UpdateCameraPosition();
  void HandleCollisions();
  void HandleOutOfBounds();
  void GenerateOutput();
  void RenderTiledMap();
  void RenderPlayer();
  int SetupEnemyGeometry();
  void RenderEnemies(int num_vertices);
  void SetupProjectileGeometry();
  void RenderProjectiles();
  void UpdateWorldOccupancyMap();
  void UpdateEnemyOccupancyMap();
  void RenderDebugWorldOccupancyMap();
  void RenderDebugEnemyOccupancyMap();
  void GetModelObservation();

  inline auto WorldToGrid(auto pos) { return (pos / kOccupancyMapResolution); }
};

}  // namespace rl2
#endif
