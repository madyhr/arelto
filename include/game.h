// include/game.h
#ifndef RL2_GAME_H_
#define RL2_GAME_H_

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <array>
#include <cstdint>
#include "constants.h"
#include "entity.h"

namespace rl2 {
// Struct to hold all the core resources
struct GameResources {
  SDL_Window* window = nullptr;
  SDL_Renderer* renderer = nullptr;
  SDL_Texture* map_texture = nullptr;
  SDL_Texture* player_texture = nullptr;
  SDL_Texture* enemy_texture = nullptr;
  SDL_Rect map_layout = {(int)0, (int)0, kWindowWidth, kWindowHeight};
};

class Camera {
public:
	Vector2D position;
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
  Enemy enemy_;
  Camera camera_;
  SDL_Vertex enemy_vertices_[kTotalEnemyVertices];
  bool is_running_;
  uint64_t ticks_count_ = 0;
  bool InitializeResources();
  bool InitializePlayer();
  bool InitializeEnemy();
  bool InitializeCamera();
  void ProcessInput();
  void Update();
  void GenerateOutput();
  void SetupEnemyGeometry();
  void UpdateEnemyPosition(float dt);
  void UpdatePlayerPosition(float dt);
  void UpdateCameraPosition();
  void HandleCollisions();
  void DetectCollisions(float dt);
  void ResolveCollisions(float dt);
};

}  // namespace rl2
#endif
