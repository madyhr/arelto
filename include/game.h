// include/game.h
#ifndef RL2_GAME_H_
#define RL2_GAME_H_

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_render.h>
#include <csignal>
#include "observation_manager.h"
#include "physics_manager.h"
#include "render_manager.h"
#include "scene.h"
#include "types.h"

namespace rl2 {

class Game {
 public:
  Game();
  ~Game();
  bool Initialize();
  void ProcessInput();
  void RunGameLoop();
  void StepGame();
  void RenderGame(float alpha);
  void ResetGame();
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
  Vector2D GetCursorPositionWorld();
  void ProcessPlayerInput();
  void CachePreviousState();
};

}  // namespace rl2
#endif
