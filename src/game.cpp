// src/game.cpp
#include "game.h"
#include <SDL2/SDL_timer.h>
#include <SDL_keycode.h>
#include <SDL_mixer.h>
#include <SDL_mouse.h>
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <iostream>
#include "audio_manager.h"
#include "constants/game.h"
#include "constants/progression_manager.h"
#include "entity.h"
#include "event_manager.h"
#include "physics_manager.h"
#include "render_manager.h"
#include "types.h"
#include "ui/widgets.h"
#include "ui_manager.h"

namespace arelto {

volatile std::sig_atomic_t Game::stop_request_ = 0;

void Game::SignalHandler(int /*signal*/) {
  stop_request_ = 1;
}

Game::Game() {};
Game::~Game() {}

bool Game::Initialize() {

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);
  game_status_.is_headless = false;

  // NOTE: Initialization order matters.
  // E.g.: Entity event subscribers should be initialized before the UI event
  // subscribers as the UI would otherwise always be 1 step behind.
  RegisterGameStateHandlers();
  entity_manager_.Initialize(event_manager_);
  item_manager_.Initialize(event_manager_);
  spell_manager_.Initialize();

  if (!(physics_manager_.Initialize())) {
    return false;
  }

  if (!(reward_manager_.Initialize())) {
    return false;
  }

  if (!(audio_manager_.Initialize())) {
    return false;
  }

  // The render manager (and thereby the UI manager) are initialized last to
  // ensure that all other handlers are called before the UI handlers are
  // dispatched to update the UI. This ensure that the UI is always up to date.
  if (!(render_manager_.Initialize(game_status_.is_headless, event_manager_,
                                   spell_manager_.GetSpellTextureMapping()))) {
    return false;
  }

  scene_.player.SetSpellManager(&spell_manager_);
  scene_.player.spell_stats_.Resize(spell_manager_.GetSpellCount());
  scene_.Reset(entity_manager_.entity_config_);
  EventContext event_context{event_manager_, scene_};
  event_manager_.DispatchImmediate(SceneResetEvent{}, event_context);
  scene_.item_archive = &item_archive_;

  if (!(Game::InitializeCamera())) {
    return false;
  }

  time_ = static_cast<float>(SDL_GetTicks64()) / 1000.0f;
  SetGameState(is_running);

  return true;
}

bool Game::InitializeCamera() {
  Vector2D player_centroid =
      GetCentroid(scene_.player.position_, scene_.player.stats_.sprite_size);
  render_manager_.camera_.UpdatePosition(player_centroid);

  return true;
};

int Game::GetGameState() {
  return static_cast<int>(game_state_);
};

void Game::SetGameState(int game_state) {
  GameState new_state = static_cast<GameState>(game_state);

  if (new_state == in_start_screen) {
    audio_manager_.StopMusic();
  } else {
    audio_manager_.PlayMusic();
  }

  // This is done in order to prevent input bleeding between states.
  is_mouse_left_active_ = false;
  is_mouse_right_active_ = false;

  previous_game_state_ = game_state_;
  game_state_ = new_state;
}

void Game::StepGamePhysics() {
  event_manager_.Flush();

  physics_manager_.StepPhysics(scene_, event_manager_);

  EventContext event_context{event_manager_, scene_};
  event_manager_.Dispatch(event_context);

  entity_manager_.Cleanup(scene_);
  obs_manager_.UpdateObservations(scene_);

  ProcessGameStateTransitionQueue();

  time_ += physics_manager_.GetPhysicsDt();
  reward_manager_.UpdateRewardTerms(scene_);
  CachePreviousState();
};

void Game::RequestGameStateTransition(GameState target) {
  for (const auto& pending : pending_transitions_) {
    if (pending == target) {
      return;
    }
  }
  pending_transitions_.push_back(target);
}

int GetGameStateTransitionPriority(GameState state) {
  switch (state) {
    case is_gameover:
      return 0;
    case in_level_up:
      return 1;
    case in_chest_opening:
      return 2;
    default:
      return 99;
  }
}

// Processes events in the game state transition queue based on the game state transition priority.
void Game::ProcessGameStateTransitionQueue() {
  if (game_state_ != is_running || pending_transitions_.empty())
    return;

  auto highest_prio_transition =
      std::min_element(pending_transitions_.begin(), pending_transitions_.end(),
                       [](GameState a, GameState b) {
                         return GetGameStateTransitionPriority(a) <
                                GetGameStateTransitionPriority(b);
                       });

  GameState next_game_state = *highest_prio_transition;
  pending_transitions_.erase(highest_prio_transition);

  switch (next_game_state) {
    case is_gameover:
      SetGameState(is_gameover);
      break;
    case in_level_up:
      SetGameState(in_level_up);
      progression_manager_.GenerateLevelUpOptions(scene_);
      render_manager_.GetUIManager().BuildLevelUpMenu(scene_.level_up_options);
      break;
    case in_chest_opening: {
      SetGameState(in_chest_opening);
      auto* root = render_manager_.GetUIManager().GetChestOpeningRoot();
      if (root) {
        auto* anim = root->FindWidgetAs<UIAnimation>("chest_animated_image");
        if (anim) {
          anim->Reset();
          anim->Play();
        }
      }
      break;
    }
    default:
      break;
  }
}

void Game::ResolveCurrentGameStateTransition() {
  SetGameState(is_running);
  ProcessGameStateTransitionQueue();
}

// Sets up all handlers for events that cause a game state transition.
void Game::RegisterGameStateHandlers() {
  event_manager_.Subscribe<PlayerDeadEvent>(
      [this](const PlayerDeadEvent&, EventContext&) {
        RequestGameStateTransition(is_gameover);
      });

  event_manager_.Subscribe<ExpGemCollectedEvent>(
      [this](const ExpGemCollectedEvent&, EventContext&) {
        if (!progression_manager_.CheckLevelUp(scene_.player))
          return;
        RequestGameStateTransition(in_level_up);
      });

  event_manager_.Subscribe<ChestOpenedEvent>(
      [this](const ChestOpenedEvent&, EventContext&) {
        RequestGameStateTransition(in_chest_opening);
      });
}

void Game::RenderGame(float alpha) {
  render_manager_.Render(scene_, alpha, game_status_, time_, game_state_);
};

void Game::ResetGame() {
  scene_.Reset(entity_manager_.entity_config_);
  EventContext event_context{event_manager_, scene_};
  event_manager_.DispatchImmediate(SceneResetEvent{}, event_context);
  item_manager_.RemoveAllItems();
  time_ = 0.0f;
  accumulator_step_ = 0.0f;
  is_mouse_left_active_ = false;
  is_mouse_right_active_ = false;
  pending_transitions_.clear();
  SetGameState(is_running);
};

// Runs the game loop continuously.
void Game::RunGameLoop() {
  float current_time = static_cast<float>(SDL_GetTicks64()) / 1000.0f;
  float accumulator = 0.0f;

  while (game_state_ != in_shutdown) {

    ProcessInput();

    float new_time = static_cast<float>(SDL_GetTicks64()) / 1000.0f;

    switch (game_state_) {

      case is_running: {
        float frame_time = new_time - current_time;
        current_time = new_time;

        // In case the frame time is too large, we override the frame time and
        // use a specified max frame time instead to avoid the "spiral of death".
        if (frame_time > kMaxFrameTime) {
          frame_time = kMaxFrameTime;
        }

        accumulator += frame_time;

        while (accumulator >= physics_manager_.GetPhysicsDt()) {
          StepGamePhysics();
          accumulator -= physics_manager_.GetPhysicsDt();
        }

        float alpha = accumulator / physics_manager_.GetPhysicsDt();

        if (game_status_.is_headless) {
          break;
        }

        entity_manager_.ProcessPendingSpawns(scene_);
        RenderGame(alpha);
        game_status_.frame_stats.print_fps_running_average(frame_time);
        break;
      }

      case in_start_screen: {
        RenderGame(0.0f);
        break;
      }

      case is_gameover:
        RenderGame(0.0f);
        break;

      case in_settings_menu: {
        current_time = new_time;

        if (game_status_.is_headless) {
          break;
        }
        RenderGame(0.0f);
        break;
      }

      case in_level_up: {
        current_time = new_time;
        RenderGame(0.0f);
        break;
      }

      case in_quit_confirm: {
        current_time = new_time;

        if (game_status_.is_headless) {
          break;
        }
        RenderGame(0.0f);
        break;
      }
      case in_chest_opening: {
        float dt = new_time - current_time;
        current_time = new_time;

        auto* chest_root = render_manager_.GetUIManager().GetChestOpeningRoot();
        if (chest_root) {
          chest_root->Update(dt);
          auto* anim =
              chest_root->FindWidgetAs<UIAnimation>("chest_animated_image");
          if (anim && anim->IsFinished()) {
            SetGameState(in_item_selection);
            progression_manager_.GenerateItemOptions(scene_);
            render_manager_.GetUIManager().BuildItemMenu(scene_.item_options);
          }
        }
        render_manager_.Render(scene_, 0.0f, game_status_, time_, game_state_);
        break;
      }
      case in_item_selection: {
        current_time = new_time;
        RenderGame(0.0f);
        break;
      }
      default:
        break;
    }
  }
};

void Game::StepGame(float dt) {
  if (game_state_ == is_running) {
    accumulator_step_ += dt;

    while (accumulator_step_ >= physics_manager_.GetPhysicsDt()) {
      StepGamePhysics();
      accumulator_step_ -= physics_manager_.GetPhysicsDt();
    }

    // ProcessPendingSpawns is called outside the accumulator loop to ensure
    // an enemy stays dead between StepGame() calls. Respawning inside the loop
    // could corrupt RL termination signals if the same enemy died, respawned,
    // and died again within a single call.
    entity_manager_.ProcessPendingSpawns(scene_);
  } else if (game_state_ == in_chest_opening) {
    auto* chest_root = render_manager_.GetUIManager().GetChestOpeningRoot();
    if (chest_root) {
      chest_root->Update(dt);
      auto* anim =
          chest_root->FindWidgetAs<UIAnimation>("chest_animated_image");
      if (anim && anim->IsFinished()) {
        SetGameState(in_item_selection);
        progression_manager_.GenerateItemOptions(scene_);
        render_manager_.GetUIManager().BuildItemMenu(scene_.item_options);
      }
    }
  }
};

void Game::ProcessInput() {

  // To be able to quit while in headless mode we need to capture ctrl+C signals
  if (stop_request_) {
    SetGameState(in_shutdown);
    std::cout << "Signal received. Exiting..." << '\n';
    return;
  };

  if (game_status_.is_headless) {
    return;
  }

  SDL_Event e;

  while (SDL_PollEvent(&e) != 0) {
    if (e.type == SDL_QUIT) {
      SetGameState(in_shutdown);
      return;
    }

    // Track mouse button releases so that stale active flags are
    // always cleared.
    if (e.type == SDL_MOUSEBUTTONUP) {
      if (e.button.button == SDL_BUTTON_LEFT) {
        is_mouse_left_active_ = false;
      }
      if (e.button.button == SDL_BUTTON_RIGHT) {
        is_mouse_right_active_ = false;
      }
    }

    if (game_state_ == in_quit_confirm) {
      ProcessQuitConfirmEvent(e);
      continue;
    }

    if (game_state_ == in_settings_menu) {
      ProcessSettingsMenuEvent(e);
      continue;
    }

    if (game_state_ == in_level_up) {
      ProcessLevelUpInput(e);
      continue;
    }
    if (game_state_ == in_chest_opening) {
      ProcessChestOpeningInput(e);
      continue;
    }
    if (game_state_ == in_item_selection) {
      ProcessItemSelectionInput(e);
      continue;
    }

    if (e.type == SDL_KEYDOWN) {
      switch (e.key.keysym.sym) {
        case SDLK_q:
          if (game_state_ == is_running) {
            SetGameState(in_quit_confirm);
          } else {
            SetGameState(in_shutdown);
            std::cout << "Key 'q' pressed! Exiting..." << '\n';
          }
          break;

        case SDLK_r:
          if (game_state_ == is_gameover) {
            ResetGame();
          }
          break;

        case SDLK_ESCAPE:
          if (game_state_ == is_running) {
            SetGameState(in_settings_menu);
          } else if (game_state_ == in_settings_menu) {
            SetGameState(is_running);
          } else {
            ResetGame();
            SetGameState(in_start_screen);
          }
          break;

        case SDLK_m:
          audio_manager_.ToggleMusic();
          break;

        case SDLK_COMMA:
          audio_manager_.DecreaseMusicVolume();
          break;

        case SDLK_PERIOD:
          audio_manager_.IncreaseMusicVolume();
          break;

        default:
          break;
      }

    } else if (e.type == SDL_MOUSEBUTTONDOWN) {
      if (game_state_ == in_start_screen) {
        int mouse_x = e.button.x;
        int mouse_y = e.button.y;

        UIManager& ui = render_manager_.GetUIManager();
        if (IsMouseOverWidget(ui.GetStartScreenRoot(), "begin_button", mouse_x,
                              mouse_y)) {
          SetGameState(is_running);
          std::cout << "Game Started!" << '\n';
        }
      } else if (game_state_ == is_running) {
        if (e.button.button == SDL_BUTTON_LEFT) {
          is_mouse_left_active_ = true;
        }
        if (e.button.button == SDL_BUTTON_RIGHT) {
          is_mouse_right_active_ = true;
        }
      }
    }
  }

  int cursor_pos_x, cursor_pos_y;
  uint32_t mouse_state = SDL_GetMouseState(&cursor_pos_x, &cursor_pos_y);

  if (game_state_ == in_settings_menu) {
    ProcessSettingsMenuInput(mouse_state);
    return;
  }

  cursor_position_ = {
      static_cast<float>(cursor_pos_x) + render_manager_.camera_.position_.x,
      static_cast<float>(cursor_pos_y) + render_manager_.camera_.position_.y};

  if (game_state_ == is_running) {
    ProcessPlayerInput();
  }
}

void Game::ProcessPlayerInput() {
  scene_.player.velocity_ = {0.0f, 0.0f};
  const Uint8* currentKeyStates = SDL_GetKeyboardState(NULL);
  if (currentKeyStates[SDL_SCANCODE_W]) {
    scene_.player.velocity_.y -= 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_S]) {
    scene_.player.velocity_.y += 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_A]) {
    scene_.player.velocity_.x -= 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_D]) {
    scene_.player.velocity_.x += 1.0f;
  }
  if (is_mouse_left_active_) {
    auto* spell = scene_.player.GetSpell(0);
    if (spell) {
      std::optional<ProjectileData> proj =
          scene_.player.CastProjectileSpell(*spell, time_, cursor_position_);
      if (proj.has_value()) {
        scene_.projectiles.AddProjectile(*proj);
      }
    }
  }
  if (is_mouse_right_active_) {
    auto* spell = scene_.player.GetSpell(1);
    if (spell) {
      std::optional<ProjectileData> proj =
          scene_.player.CastProjectileSpell(*spell, time_, cursor_position_);
      if (proj.has_value()) {
        scene_.projectiles.AddProjectile(*proj);
      }
    }
  }
}

void Game::ProcessSettingsMenuEvent(const SDL_Event& e) {
  if (e.type == SDL_KEYDOWN) {
    switch (e.key.keysym.sym) {
      case SDLK_q:
        SetGameState(in_quit_confirm);
        std::cout << "Quit confirmation requested..." << '\n';
        break;
      case SDLK_ESCAPE:
        SetGameState(is_running);
        break;
      case SDLK_m:
        audio_manager_.ToggleMusic();
        break;
      case SDLK_COMMA:
        audio_manager_.DecreaseMusicVolume();
        break;
      case SDLK_PERIOD:
        audio_manager_.IncreaseMusicVolume();
        break;
      default:
        break;
    }
  } else if (e.type == SDL_MOUSEBUTTONDOWN) {
    int mouse_x = e.button.x;
    int mouse_y = e.button.y;

    UIManager& ui = render_manager_.GetUIManager();
    UIWidget* settings_root = ui.GetSettingsRoot();

    if (IsMouseOverWidget(settings_root, "mute_checkbox", mouse_x, mouse_y)) {
      audio_manager_.ToggleMusic();
    }

    if (IsMouseOverWidget(settings_root, "main_menu_button", mouse_x,
                          mouse_y)) {
      ResetGame();
      SetGameState(in_start_screen);
    }

    if (IsMouseOverWidget(settings_root, "resume_button", mouse_x, mouse_y)) {
      SetGameState(is_running);
    }

    if (IsMouseOverWidget(settings_root, "occupancy_map_checkbox", mouse_x,
                          mouse_y)) {
      game_status_.show_occupancy_map = !game_status_.show_occupancy_map;
    }

    if (IsMouseOverWidget(settings_root, "ray_caster_checkbox", mouse_x,
                          mouse_y)) {
      game_status_.show_ray_caster = !game_status_.show_ray_caster;
    }
  }
}

void Game::ProcessQuitConfirmEvent(const SDL_Event& e) {
  if (e.type == SDL_KEYDOWN) {
    switch (e.key.keysym.sym) {
      case SDLK_y:
        SetGameState(in_shutdown);
        std::cout << "Quit confirmed. Exiting..." << '\n';
        break;
      case SDLK_ESCAPE:
      case SDLK_n:
        SetGameState(previous_game_state_);
        break;
      default:
        break;
    }
  } else if (e.type == SDL_MOUSEBUTTONDOWN) {
    int mouse_x = e.button.x;
    int mouse_y = e.button.y;

    UIManager& ui = render_manager_.GetUIManager();
    UIWidget* quit_root = ui.GetQuitConfirmRoot();

    if (IsMouseOverWidget(quit_root, "quit_yes_button", mouse_x, mouse_y)) {
      SetGameState(in_shutdown);
      std::cout << "Quit confirmed. Exiting..." << '\n';
    }

    if (IsMouseOverWidget(quit_root, "quit_no_button", mouse_x, mouse_y)) {
      SetGameState(previous_game_state_);
    }
  }
}

void Game::ProcessSettingsMenuInput(uint32_t mouse_state) {
  if (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
    int mouse_x, mouse_y;
    SDL_GetMouseState(&mouse_x, &mouse_y);

    UIManager& ui = render_manager_.GetUIManager();
    UIWidget* slider = ui.GetSettingsRoot()->FindWidget("volume_slider");
    if (slider) {
      SDL_Rect slider_bounds = slider->GetComputedBounds();
      // A little extra padding is added to y to make it more user-friendly.
      if (mouse_x >= slider_bounds.x &&
          mouse_x <= slider_bounds.x + slider_bounds.w &&
          mouse_y >= slider_bounds.y - 10 &&
          mouse_y <= slider_bounds.y + slider_bounds.h + 10) {
        float percent = static_cast<float>(mouse_x - slider_bounds.x) /
                        static_cast<float>(slider_bounds.w);
        audio_manager_.SetMusicVolume(percent);
      }
    }
  }

  // The SDL_mixer music volume goes from 0 to 128.
  render_manager_.UpdateSettingsMenuState(
      audio_manager_.GetMusicVolume() * 128.0f, audio_manager_.IsMusicMuted(),
      game_status_);
}

// This function processes inputs during a level up state and applies the
// upgrade option according to which button was clicked.
void Game::ProcessLevelUpInput(const SDL_Event& e) {
  if (e.type == SDL_KEYDOWN) {
    switch (e.key.keysym.sym) {
      case SDLK_q:
        SetGameState(in_quit_confirm);
        break;
      default:
        break;
    }
  } else if (e.type == SDL_MOUSEBUTTONDOWN &&
             e.button.button == SDL_BUTTON_LEFT) {
    int mouse_x = e.button.x;
    int mouse_y = e.button.y;

    UIManager& ui = render_manager_.GetUIManager();

    for (int i = 0; i < kNumSpellUpgradeOptions; ++i) {
      std::string btn_id = "select_button_" + std::to_string(i);

      if (IsMouseOverWidget(ui.GetLevelUpRoot(), btn_id, mouse_x, mouse_y)) {
        progression_manager_.ApplyLevelUpUpgrade(scene_, i);
        EventContext event_context{event_manager_, scene_};
        event_manager_.DispatchImmediate(PlayerLevelUpEvent{}, event_context);
        UIWidget* menu = ui.GetLevelUpRoot();
        if (menu) {
          menu->SetVisible(false);
        }
        ResolveCurrentGameStateTransition();
        return;
      }
    }
  }
}

void Game::ProcessChestOpeningInput(const SDL_Event& e) {
  // During the animation phase, only allow quitting
  if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_q) {
    SetGameState(in_quit_confirm);
    return;
  }
  return;
}

void Game::ProcessItemSelectionInput(const SDL_Event& e) {
  if (e.type == SDL_KEYDOWN) {
    switch (e.key.keysym.sym) {
      case SDLK_q:
        SetGameState(in_quit_confirm);
        break;
      default:
        break;
    }
  } else if (e.type == SDL_MOUSEBUTTONDOWN &&
             e.button.button == SDL_BUTTON_LEFT) {
    int mouse_x = e.button.x;
    int mouse_y = e.button.y;

    UIManager& ui = render_manager_.GetUIManager();

    for (int i = 0; i < kNumItemOptions; ++i) {
      std::string btn_id = "select_button_" + std::to_string(i);

      if (IsMouseOverWidget(ui.GetItemMenuRoot(), btn_id, mouse_x, mouse_y)) {
        progression_manager_.ApplyItemUpgrade(scene_, item_manager_, i);
        UIWidget* menu = ui.GetItemMenuRoot();
        if (menu) {
          menu->SetVisible(false);
        }
        EventContext event_context{event_manager_, scene_};
        event_manager_.DispatchImmediate(PlayerClaimedItemEvent{},
                                         event_context);
        ResolveCurrentGameStateTransition();
        return;
      }
    }
  }
}

void Game::CachePreviousState() {
  scene_.player.prev_position_ = scene_.player.position_;

  for (int i = 0; i < kNumEnemies; ++i) {
    if (scene_.enemy.is_alive[i]) {
      scene_.enemy.prev_position[i] = scene_.enemy.position[i];
      scene_.enemy.prev_velocity[i] = scene_.enemy.velocity[i];
    }
  }

  size_t num_proj = scene_.projectiles.GetNumProjectiles();
  for (size_t i = 0; i < num_proj; ++i) {
    scene_.projectiles.prev_position_[i] = scene_.projectiles.position_[i];
  }

  size_t num_gems = scene_.exp_gem.GetNumExpGems();
  for (size_t i = 0; i < num_gems; ++i) {
    scene_.exp_gem.prev_position_[i] = scene_.exp_gem.position_[i];
  }

  size_t num_chests = scene_.chest.GetNumChests();
  for (size_t i = 0; i < num_chests; ++i) {
    scene_.chest.prev_position_[i] = scene_.chest.position_[i];
  }

  render_manager_.camera_.prev_position_ = render_manager_.camera_.position_;
}

void Game::Shutdown() {
  render_manager_.Shutdown();
  audio_manager_.Shutdown();
}

// Function for checking whether the mouse cursor is currently on top of a
// UIWidget with a given root and id.
bool Game::IsMouseOverWidget(UIWidget* root, const std::string& widget_id,
                             int mouse_x, int mouse_y) {
  if (!root) {
    return false;
  }
  UIWidget* widget = root->FindWidget(widget_id);
  if (!widget) {
    return false;
  }
  SDL_Rect bounds = widget->GetComputedBounds();
  return mouse_x >= bounds.x && mouse_x <= bounds.x + bounds.w &&
         mouse_y >= bounds.y && mouse_y <= bounds.y + bounds.h;
}

}  // namespace arelto
