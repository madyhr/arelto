// src/game.cpp
#include <SDL2/SDL_timer.h>
#include <iostream>
#include "game.h"
#include "constants.h"

namespace rl2 {

Game::Game() {
};

Game::~Game() {
	Game::Shutdown();
}

bool Game::Initialize() {

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::cerr << "SDL could not initialize! SDL Error: " << SDL_GetError() << std::endl;
		return false;
	}

	resources.window = SDL_CreateWindow(
		"RL2",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		kWindowWidth,
		kWindowHeight,
		SDL_WINDOW_SHOWN
	);

	if (resources.window == nullptr) {
		std::cerr << "Window could not be created: " << SDL_GetError() << std::endl;
		return false;
	}

	resources.renderer = SDL_CreateRenderer(
		resources.window, 
		-1, 
		SDL_RENDERER_ACCELERATED
	);

	if (resources.renderer == nullptr) {
		std::cerr << "Renderer could not be created: " << SDL_GetError() << std::endl;
		return false;
	}

	int imgFlags = IMG_INIT_PNG;
		if (!(IMG_Init(imgFlags) & imgFlags)) {
		std::cerr << "SDL Images could not be initialized: " << SDL_GetError() << std::endl;
		return false;
	}

	resources.map_texture = IMG_LoadTexture(resources.renderer, "assets/textures/grassy_plains.png");
	resources.player_texture = IMG_LoadTexture(resources.renderer, "assets/textures/wizard.png");
	resources.enemy_texture = IMG_LoadTexture(resources.renderer, "assets/textures/goblin.png");

	if (resources.map_texture == nullptr || resources.player_texture == nullptr || resources.enemy_texture == nullptr) {
		std::cerr << "One or more textures could not be loaded: " << SDL_GetError() << std::endl;
		return false;
	}

	// Initialize Player
	player_.position = {kPlayerInitX, kPlayerInitY};
	player_.stats.movement_speed = kPlayerSpeed;
	player_.stats.size = {kPlayerWidth, kPlayerHeight};

	// Initialize Enemy
	enemy_.position = {kEnemyInitX, kEnemyInitY};
	enemy_.stats.movement_speed = kEnemySpeed;
	enemy_.stats.size = {kEnemyWidth, kEnemyHeight};

	// Initialize game state
	resources.map_layout = {0, 0, kWindowWidth, kWindowHeight};
	ticks_count_ = SDL_GetTicks();
	is_running_ = true;

	return true;
}


void Game::RunGameLoop() {
	while (is_running_){
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
		}
		else if (e.type == SDL_KEYDOWN){

			switch (e.key.keysym.sym){
				case SDLK_q:
					is_running_ = false;
					std::cout << "Key 'q' pressed! Exiting..." << std::endl;
					break;
			}
		}
	}

	player_.velocity = {0.0f, 0.0f};
	const Uint8* currentKeyStates = SDL_GetKeyboardState(NULL);
	if (currentKeyStates[SDL_SCANCODE_W]) { player_.velocity.y-= 1.0f; }
	if (currentKeyStates[SDL_SCANCODE_S]) { player_.velocity.y+= 1.0f; }
	if (currentKeyStates[SDL_SCANCODE_A]) { player_.velocity.x-= 1.0f; }
	if (currentKeyStates[SDL_SCANCODE_D]) { player_.velocity.x+= 1.0f; }

}

void Game::UpdateGame() {
	float delta_time = (SDL_GetTicks() - ticks_count_) / 1000.0f;

	ticks_count_ = SDL_GetTicks();

	// Update Player
	float player_velocity_magnitude = std::hypot(player_.velocity.x, player_.velocity.y);
	if (player_velocity_magnitude > 1.0f) {
		player_.velocity.x /= player_velocity_magnitude;
		player_.velocity.y /= player_velocity_magnitude;
	}
	player_.position.x += player_.velocity.x * player_.stats.movement_speed * delta_time;
	player_.position.y += player_.velocity.y * player_.stats.movement_speed * delta_time;

	// Update Enemy
	float distance_to_player = std::hypot(player_.position.x - enemy_.position.x, player_.position.y - enemy_.position.y);
	enemy_.velocity.x = (player_.position.x - enemy_.position.x) / distance_to_player;
	enemy_.velocity.y = (player_.position.y - enemy_.position.y) / distance_to_player;
	enemy_.position.x += enemy_.velocity.x * enemy_.stats.movement_speed * delta_time;
	enemy_.position.y += enemy_.velocity.y * enemy_.stats.movement_speed * delta_time;

};

void Game::GenerateOutput() {
	SDL_SetRenderDrawColor(resources.renderer, 0x00, 0x00, 0x00, 0xFF);
	SDL_RenderClear(resources.renderer);
	SDL_RenderCopy(resources.renderer, resources.map_texture, NULL, &resources.map_layout);
	SDL_Rect player_render_box = {
		(int)player_.position.x,
		(int)player_.position.y,
		(int)player_.stats.size.width,
		(int)player_.stats.size.height
	};
	SDL_RenderCopy(resources.renderer, resources.player_texture, NULL, &player_render_box);
	SDL_Rect enemy_render_box = {
		(int)enemy_.position.x,
		(int)enemy_.position.y,
		(int)enemy_.stats.size.width,
		(int)enemy_.stats.size.height
	};
	SDL_RenderCopy(resources.renderer, resources.enemy_texture, NULL, &enemy_render_box);
	SDL_RenderPresent(resources.renderer);
};


void Game::Shutdown() {

    if (resources.map_texture) {
        SDL_DestroyTexture(resources.map_texture);
        resources.map_texture = nullptr;
    }
    if (resources.player_texture) {
        SDL_DestroyTexture(resources.player_texture);
        resources.player_texture = nullptr;
    }
    if (resources.enemy_texture) {
        SDL_DestroyTexture(resources.enemy_texture);
        resources.enemy_texture = nullptr;
    }

    IMG_Quit();

    if (resources.renderer) {
        SDL_DestroyRenderer(resources.renderer);
        resources.renderer = nullptr;
    }

    if (resources.window) {
        SDL_DestroyWindow(resources.window);
        resources.window = nullptr;
    }

    SDL_Quit();
}


} // namespace rl2

