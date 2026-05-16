// include/game.h
#ifndef RL2_GAME_H_
#define RL2_GAME_H_

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_mixer.h>
#include <SDL2/SDL_render.h>
#include <SDL_events.h>
#include <csignal>
#include <vector>
#include "action_manager.h"
#include "audio_manager.h"
#include "entity_manager.h"
#include "event_manager.h"
#include "item_manager.h"
#include "items.h"
#include "observation_manager.h"
#include "physics_manager.h"
#include "progression_manager.h"
#include "render_manager.h"
#include "reward_manager.h"
#include "scene.h"
#include "spell_manager.h"
#include "types.h"

namespace arelto {

class Game {
 public:
  Game();
  ~Game();
  bool Initialize();
  void ProcessInput();
  void RunGameLoop();
  void StepGame(float dt);
  void RenderGame(float alpha);
  void ResetGame();
  void Shutdown();
  int GetGameState();
  void SetGameState(GameState game_state);
  static void SignalHandler(int signal);

  Scene scene_;
  ObservationManager obs_manager_;
  ActionManager action_manager_;
  RewardManager reward_manager_;

 private:
  RenderManager render_manager_;
  PhysicsManager physics_manager_;
  EntityManager entity_manager_;
  ProgressionManager progression_manager_;
  SpellManager spell_manager_;
  AudioManager audio_manager_;
  GameStatus game_status_;
  GameState game_state_;
  Vector2D cursor_position_;
  float time_ = 0.0f;
  float accumulator_step_;
  GameState previous_game_state_ = GameState::in_start_screen;
  bool is_mouse_left_active_ = false;
  bool is_mouse_right_active_ = false;
  std::vector<GameState> pending_transitions_;
  ItemArchive item_archive_;
  EventManager event_manager_;
  ItemManager item_manager_;

  static volatile std::sig_atomic_t stop_request_;
  bool InitializeCamera();
  void RegisterGameStateHandlers();
  void RequestGameStateTransition(GameState target);
  void ProcessGameStateTransitionQueue();
  void ResolveCurrentGameStateTransition();
  void StepGamePhysics();
  Vector2D GetCursorPositionWorld();
  void ProcessPlayerInput();
  void ProcessLevelUpInput(const SDL_Event& e);
  void ProcessItemSelectionInput(const SDL_Event& e);
  void ProcessSettingsMenuInput(uint32_t mouse_state);
  void ProcessSettingsMenuEvent(const SDL_Event& e);
  void ProcessQuitConfirmEvent(const SDL_Event& e);
  void ProcessChestOpeningInput(const SDL_Event& e);
  void CachePreviousState();
  bool IsMouseOverWidget(UIWidget* root, const std::string& widget_id,
                         int mouse_x, int mouse_y);
};

}  // namespace arelto
#endif
