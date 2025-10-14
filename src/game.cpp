// src/game.cpp
#include "game.h"
#include <SDL2/SDL_timer.h>
#include <SDL_render.h>
#include <iostream>
#include "constants.h"
#include "random.h"

namespace rl2 {

Game::Game(){};

Game::~Game() {
  Game::Shutdown();
}

bool Game::Initialize() {

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

  // Initialize Player
  player_.position = {kPlayerInitX, kPlayerInitY};
  player_.stats.movement_speed = kPlayerSpeed;
  player_.stats.size = {kPlayerWidth, kPlayerHeight};

  // Initialize Enemies
  std::fill(enemy_.movement_speed.begin(), enemy_.movement_speed.end(),
            kEnemySpeed);
  std::fill(enemy_.size.begin(), enemy_.size.end(),
            Size{kEnemyHeight, kEnemyWidth});
  std::fill(enemy_.health.begin(), enemy_.health.end(), kEnemyHealth);

  for (int i = 0; i < kNumEnemies; ++i) {
    enemy_.position[i] = {kEnemyInitX + i * 50, kEnemyInitY + i * 50};
    enemy_.movement_speed[i] += generate_random_int(1, 100);
  };

  resources_.map_layout = {0, 0, kWindowWidth, kWindowHeight};
  ticks_count_ = SDL_GetTicks();
  is_running_ = true;

  return true;
}

void Game::RunGameLoop() {
  while (is_running_) {
    Game::ProcessInput();
    Game::UpdateGame();
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

void Game::UpdateGame() {
  float delta_time = (SDL_GetTicks() - ticks_count_) / 1000.0f;

  ticks_count_ = SDL_GetTicks();

  // Update Player
  float player_velocity_magnitude =
      std::hypot(player_.velocity.x, player_.velocity.y);
  if (player_velocity_magnitude > 1.0f) {
    player_.velocity.x /= player_velocity_magnitude;
    player_.velocity.y /= player_velocity_magnitude;
  }
  player_.position.x +=
      player_.velocity.x * player_.stats.movement_speed * delta_time;
  player_.position.y +=
      player_.velocity.y * player_.stats.movement_speed * delta_time;

  // Update Enemy
  for (int i = 0; i < kNumEnemies; ++i) {
    float distance_to_player =
        std::hypot(player_.position.x - enemy_.position[i].x,
                   player_.position.y - enemy_.position[i].y);
    enemy_.velocity[i].x =
        (player_.position.x - enemy_.position[i].x) / distance_to_player;
    enemy_.velocity[i].y =
        (player_.position.y - enemy_.position[i].y) / distance_to_player;
    enemy_.position[i].x +=
        enemy_.velocity[i].x * enemy_.movement_speed[i] * delta_time;
    enemy_.position[i].y +=
        enemy_.velocity[i].y * enemy_.movement_speed[i] * delta_time;
  }
};

void Game::GenerateOutput() {
  SDL_SetRenderDrawColor(resources_.renderer, 0x00, 0x00, 0x00, 0xFF);
  SDL_RenderClear(resources_.renderer);
  SDL_RenderCopy(resources_.renderer, resources_.map_texture, NULL,
                 &resources_.map_layout);
  SDL_Rect player_render_box = {
      (int)player_.position.x, (int)player_.position.y,
      (int)player_.stats.size.width, (int)player_.stats.size.height};
  SDL_RenderCopy(resources_.renderer, resources_.player_texture, NULL,
                 &player_render_box);

  SetupEnemyGeometry();
  SDL_RenderGeometry(resources_.renderer, resources_.enemy_texture,
                     enemy_vertices_, kTotalEnemyVertices, nullptr, 0);

  SDL_RenderPresent(resources_.renderer);
};

void Game::SetupEnemyGeometry() {
  for (int i = 0; i < kNumEnemies; ++i) {
    float x = enemy_.position[i].x;
    float y = enemy_.position[i].y;
    float w = (float)kEnemyWidth;
    float h = (float)kEnemyHeight;

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
