// include/game.h
#ifndef RL2_GAME_H_
#define RL2_GAME_H_

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_render.h>
#include <array>
#include <csignal>
#include "constants.h"
#include "observation_manager.h"
#include "physics_manager.h"
#include "render_manager.h"
#include "scene.h"
#include "types.h"

namespace rl2 {

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
  bool is_debug;
  bool is_headless;
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
  void StepGame();
  void RenderGame(float alpha);
  void Shutdown();
  int GetGameState();
  void SetGameState(int game_state);
  static void SignalHandler(int signal);

  Scene scene_;
  ObservationManager obs_manager_;

 private:
  RenderManager render_manager_;
  PhysicsManager physics_manager_;
  GameStatus game_status_;
  GameState game_state_;
  Vector2D cursor_position_;
  float time_ = 0.0f;
  static volatile std::sig_atomic_t stop_request_;
  bool InitializeCamera();
  void ProcessInput();
  Vector2D GetCursorPositionWorld();
  void ProcessPlayerInput();
  void CachePreviousState();
};

}  // namespace rl2
#endif
