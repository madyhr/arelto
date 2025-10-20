// src/game.cpp
#include "game.h"
#include <SDL2/SDL_timer.h>
#include <SDL_render.h>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include "constants.h"
#include "game_math.h"
#include "random.h"

namespace rl2 {

Game::Game(){};

Game::~Game() {
  Game::Shutdown();
}

bool Game::Initialize() {
  if (!(Game::InitializeResources())) {
    return true;
  }
  if (!(Game::InitializePlayer())) {
    return true;
  }
  if (!(Game::InitializeEnemy())) {
    return true;
  }

  ticks_count_ = SDL_GetTicks64();
  is_running_ = true;

  return true;
}

bool Game::InitializeResources() {

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL could not initialize! SDL Error: " << SDL_GetError()
              << std::endl;
    return false;
  }

  resources_.window =
      SDL_CreateWindow("RL2", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                       kWindowWidth, kWindowHeight, SDL_WINDOW_SHOWN);

  if (resources_.window == nullptr) {
    std::cerr << "Window could not be created: " << SDL_GetError() << std::endl;
    return false;
  }

  resources_.renderer =
      SDL_CreateRenderer(resources_.window, -1, SDL_RENDERER_ACCELERATED);

  if (resources_.renderer == nullptr) {
    std::cerr << "Renderer could not be created: " << SDL_GetError()
              << std::endl;
    return false;
  }

  int imgFlags = IMG_INIT_PNG;
  if (!(IMG_Init(imgFlags) & imgFlags)) {
    std::cerr << "SDL Images could not be initialized: " << SDL_GetError()
              << std::endl;
    return false;
  }

  resources_.map_texture =
      IMG_LoadTexture(resources_.renderer, "assets/textures/grassy_plains.png");
  resources_.player_texture =
      IMG_LoadTexture(resources_.renderer, "assets/textures/wizard.png");
  resources_.enemy_texture =
      IMG_LoadTexture(resources_.renderer, "assets/textures/goblin.png");

  if (resources_.map_texture == nullptr ||
      resources_.player_texture == nullptr ||
      resources_.enemy_texture == nullptr) {
    std::cerr << "One or more textures could not be loaded: " << SDL_GetError()
              << std::endl;
    return false;
  }

  resources_.map_layout = {0, 0, kWindowWidth, kWindowHeight};
  return true;
};

bool Game::InitializePlayer() {
  player_.stats.size = {kPlayerWidth, kPlayerHeight};
  player_.position = {kPlayerInitX, kPlayerInitY};
  player_.stats.movement_speed = kPlayerSpeed;
  return true;
};

bool Game::InitializeEnemy() {
  std::fill(enemy_.movement_speed.begin(), enemy_.movement_speed.end(),
            kEnemySpeed);
  std::fill(enemy_.size.begin(), enemy_.size.end(),
            Size{kEnemyHeight, kEnemyWidth});
  std::fill(enemy_.health.begin(), enemy_.health.end(), kEnemyHealth);

  int max_x = kWindowWidth - kEnemyWidth;
  int max_y = kWindowHeight - kEnemyHeight;

  for (int i = 0; i < kNumEnemies; ++i) {
    Vector2D potential_pos;

    do {
      potential_pos = {(float)generate_random_int(0, max_x),
                       (float)generate_random_int(0, max_y)};

    } while (calculate_distance_vector2d(potential_pos, player_.position) <
             kEnemyMinimumInitialDistance);

    enemy_.position[i] = potential_pos;
    enemy_.movement_speed[i] += generate_random_int(1, 100);
    enemy_.size[i].height += generate_random_int(1, 50);
    enemy_.size[i].width += generate_random_int(1, 50);
  };

  SDL_Color red = {255, 0, 0, 255};
  for (int i = 0; i < kTotalEnemyVertices; ++i) {
    enemy_vertices_[i].color = red;
  }
  return true;
};

void FrameStats::update_frame_time_buffer(float new_value) {
  float oldest_value = frame_time_buffer[head_index];
  frame_time_sum = frame_time_sum - oldest_value + new_value;
  frame_time_buffer[head_index] = new_value;

  if (current_buffer_length < max_buffer_length) {
    current_buffer_length++;
  };

  head_index = (head_index + 1) % max_buffer_length;
};

float FrameStats::get_average_frame_time() {
  if (current_buffer_length == 0) {
    return 0.0f;
  }
  return static_cast<float>(frame_time_sum / current_buffer_length);
};

void FrameStats::print_fps_running_average(float dt) {
  static float accumulated_time = 0.0f;
  update_frame_time_buffer(dt);
  if (accumulated_time > 1.0f) {
    float avg_frame_time = get_average_frame_time();
    float average_fps = 1.0f / (avg_frame_time);
    std::cout << "Current Avg FPS: " << std::fixed << average_fps;
    std::cout << "\r" << std::flush;
    accumulated_time -= 1.0f;
  };
  accumulated_time += dt;
};

void Game::RunGameLoop() {
  while (is_running_) {
    Game::ProcessInput();
    Game::Update();
    Game::GenerateOutput();
  }
};

void Game::ProcessInput() {

  SDL_Event e;

  while (SDL_PollEvent(&e) != 0) {
    if (e.type == SDL_QUIT) {
      is_running_ = false;
    } else if (e.type == SDL_KEYDOWN) {

      switch (e.key.keysym.sym) {
        case SDLK_q:
          is_running_ = false;
          std::cout << "Key 'q' pressed! Exiting..." << std::endl;
          break;
      }
    }
  }

  player_.velocity = {0.0f, 0.0f};
  const Uint8* currentKeyStates = SDL_GetKeyboardState(NULL);
  if (currentKeyStates[SDL_SCANCODE_W]) {
    player_.velocity.y -= 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_S]) {
    player_.velocity.y += 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_A]) {
    player_.velocity.x -= 1.0f;
  }
  if (currentKeyStates[SDL_SCANCODE_D]) {
    player_.velocity.x += 1.0f;
  }
}

void Game::Update() {
  Uint64 current_ticks = SDL_GetTicks64();
  float dt = (float)(current_ticks - ticks_count_) / 1000.0f;

  Game::UpdatePlayerPosition(dt);
  Game::UpdateEnemyPosition(dt);
  Game::HandleCollisions();

  game_status_.frame_stats.print_fps_running_average(dt);

  ticks_count_ = current_ticks;
}
void Game::HandleCollisions() {
  rl2::handle_collisions_sap(player_, enemy_);
};

void Game::UpdatePlayerPosition(float dt) {

  float player_velocity_magnitude = get_length_vector2d(player_.velocity);
  if (player_velocity_magnitude > 1.0f) {
    player_.velocity.x /= player_velocity_magnitude;
    player_.velocity.y /= player_velocity_magnitude;
  }

  player_.position.x += player_.velocity.x * player_.stats.movement_speed * dt;
  player_.position.y += player_.velocity.y * player_.stats.movement_speed * dt;
};

void Game::UpdateEnemyPosition(float dt) {
  for (int i = 0; i < kNumEnemies; ++i) {
    float dx = player_.position.x - enemy_.position[i].x;
    float dy = player_.position.y - enemy_.position[i].y;
    float distance_to_player = std::hypot(dx, dy);
    enemy_.velocity[i].x = dx / (distance_to_player + 1e-6);
    enemy_.velocity[i].y = dy / (distance_to_player + 1e-6);
    enemy_.position[i].x +=
        enemy_.velocity[i].x * enemy_.movement_speed[i] * dt;
    enemy_.position[i].y +=
        enemy_.velocity[i].y * enemy_.movement_speed[i] * dt;
  }
};

void Game::GenerateOutput() {
  SDL_SetRenderDrawColor(resources_.renderer, 0x00, 0x00, 0x00, 0xFF);
  SDL_RenderClear(resources_.renderer);
  SDL_RenderCopy(resources_.renderer, resources_.map_texture, NULL,
                 &resources_.map_layout);
  SDL_Rect player_render_box = {
      (int)(player_.position.x), (int)(player_.position.y),
      (int)player_.stats.size.width, (int)player_.stats.size.height};

  SDL_RenderCopy(resources_.renderer, resources_.player_texture, NULL,
                 &player_render_box);
  SDL_RenderGeometry(resources_.renderer, resources_.enemy_texture,
                     enemy_vertices_, kTotalEnemyVertices, nullptr, 0);
  // For debugging render boxes
  // SDL_SetRenderDrawColor(resources_.renderer, 0, 0, 0, 255);
  // SDL_RenderFillRect(resources_.renderer, &player_render_box);
  // SDL_RenderGeometry(resources_.renderer, nullptr, enemy_vertices_,
  //                    kTotalEnemyVertices, nullptr, 0);
  //
  SetupEnemyGeometry();
  SDL_RenderPresent(resources_.renderer);
};

void Game::SetupEnemyGeometry() {
  for (int i = 0; i < kNumEnemies; ++i) {
    float x = enemy_.position[i].x;
    float y = enemy_.position[i].y;
    float w = enemy_.size[i].width;
    float h = enemy_.size[i].height;

    int vertex_offset = i * kEnemyVertices;

    // --- Vertices for Triangle 1 (Top-Left, Bottom-Left, Bottom-Right) ---
    // 1. Top-Left
    enemy_vertices_[vertex_offset + 0] = {
        {x, y}, {255, 255, 255, 255}, {kTexCoordLeft, kTexCoordTop}};
    // 2. Bottom-Left
    enemy_vertices_[vertex_offset + 1] = {
        {x, y + h}, {255, 255, 255, 255}, {kTexCoordLeft, kTexCoordBottom}};
    // 3. Bottom-Right
    enemy_vertices_[vertex_offset + 2] = {{x + w, y + h},
                                          {255, 255, 255, 255},
                                          {kTexCoordRight, kTexCoordBottom}};
    // --- Vertices for Triangle 2 (Top-Left, Bottom-Right, Top-Right) ---
    // 4. Top-Left (Repeat)
    enemy_vertices_[vertex_offset + 3] =
        enemy_vertices_[vertex_offset + 0];  // Same as vertex 1
    // 5. Bottom-Right (Repeat)
    enemy_vertices_[vertex_offset + 4] =
        enemy_vertices_[vertex_offset + 2];  // Same as vertex 3
    // 6. Top-Right
    enemy_vertices_[vertex_offset + 5] = {
        {x + w, y}, {255, 255, 255, 255}, {kTexCoordRight, kTexCoordTop}};
  }
};

void Game::Shutdown() {

  if (resources_.map_texture) {
    SDL_DestroyTexture(resources_.map_texture);
    resources_.map_texture = nullptr;
  }
  if (resources_.player_texture) {
    SDL_DestroyTexture(resources_.player_texture);
    resources_.player_texture = nullptr;
  }
  if (resources_.enemy_texture) {
    SDL_DestroyTexture(resources_.enemy_texture);
    resources_.enemy_texture = nullptr;
  }

  IMG_Quit();

  if (resources_.renderer) {
    SDL_DestroyRenderer(resources_.renderer);
    resources_.renderer = nullptr;
  }

  if (resources_.window) {
    SDL_DestroyWindow(resources_.window);
    resources_.window = nullptr;
  }

  SDL_Quit();
}

}  // namespace rl2
