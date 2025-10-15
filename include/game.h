// include/game.h
#ifndef RL2_GAME_H_
#define RL2_GAME_H_

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
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

class Game {
 public:
  Game();
  ~Game();
  bool Initialize();
  void RunGameLoop();
  void Shutdown();

 private:
  GameResources resources_;
  Player player_;
  Enemy enemy_;
  SDL_Vertex enemy_vertices_[kTotalEnemyVertices];
  void ProcessInput();
  void UpdateGame();
  void GenerateOutput();
  void SetupEnemyGeometry();
  void UpdateEnemyPosition(float);
  void UpdatePlayerPosition(float);

  bool is_running_;
  uint32_t ticks_count_ = 0;
};

}  // namespace rl2
#endif
