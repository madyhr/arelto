#include <SDL2/SDL_error.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_keyboard.h>
#include <SDL2/SDL_rect.h>
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_image.h>
#include <iostream>
#include <SDL2/SDL.h>
#include <cmath>

// Player constants
const float kPlayerInitX = 400.0f;
const float kPlayerInitY = 300.0f;
const float kPlayerSpeed = 200.0f;
const int kPlayerHeight = 75;
const int kPlayerWidth = 75;
// Enemy constants
const float kEnemyInitX = 100.0f;
const float kEnemyInitY = 100.0f;
const float kEnemySpeed = 40.0f;
const int kEnemyHeight = 50;
const int kEnemyWidth = 50;


const int kWindowHeight = 600;
const int kWindowWidth = 800;

struct Position {
	float x;
	float y;
};

Position playerState = {kPlayerInitX, kPlayerInitY};
Position enemyState = {kEnemyInitX, kEnemyInitY};

struct StateChange {
	float x;
	float y;
};


int main(int argc, char* args[]){

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::cerr << "SDL could not initialize! SDL Error: " << SDL_GetError() << std::endl;
		return 1;
	}

	SDL_Window* window = SDL_CreateWindow(
		"My_CPP_Game",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		kWindowWidth,
		kWindowHeight,
		SDL_WINDOW_SHOWN
	);

	if (window == NULL){
		std::cerr << "Window could not be created: " << SDL_GetError() << std::endl;
		SDL_Quit();
		return 1;
	}

	SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

	if (renderer == NULL) {
		std::cerr << "Renderer could not be created: " << SDL_GetError() << std::endl;
		SDL_Quit();
		return 1;
	}

	int imgFlags = IMG_INIT_PNG;
	if (!(IMG_Init(imgFlags) & imgFlags)){
		std::cerr << "SDL Images could not be initialized: " << SDL_GetError() << std::endl;
		return 1;
	}	

	SDL_Texture* playerTexture = IMG_LoadTexture(renderer, "wizard.png");
	SDL_Texture* enemyTexture = IMG_LoadTexture(renderer, "goblin.png");

	if (playerTexture == NULL) {
		std::cerr << "Texture could not be loaded: " << SDL_GetError() << std::endl;
		SDL_DestroyRenderer(renderer);
		SDL_DestroyWindow(window);
		IMG_Quit();
		SDL_Quit();
		return 1;
	}	

	if (enemyTexture == NULL) {
		std::cerr << "Texture could not be loaded: " << SDL_GetError() << std::endl;
		SDL_DestroyRenderer(renderer);
		SDL_DestroyWindow(window);
		IMG_Quit();
		SDL_Quit();
		return 1;
	}	

	SDL_Rect playerHitBox = {
		(int)playerState.x, // x pos
		(int)playerState.y, // y pos
		kPlayerWidth, // width
		kPlayerHeight// height
	};

	SDL_Rect enemyHitBox = {
		(int)enemyState.x, // x pos
		(int)enemyState.y, // y pos
		kEnemyWidth, // width
		kEnemyHeight// height
	};

	bool quit = false;
	SDL_Event e;

	Uint32 previous_time = SDL_GetTicks();

	while (!quit){

		Uint32 current_time = SDL_GetTicks(); 
		float dt = (current_time - previous_time)/1000.0f;

		while (SDL_PollEvent(&e) != 0) {
			if (e.type == SDL_QUIT) {
				quit = true;
			}
			else if (e.type == SDL_KEYDOWN){

				switch (e.key.keysym.sym){
					case SDLK_q:
						quit = true;
						std::cout << "Key 'q' pressed! Exiting..." << std::endl;
						break;
				}
			}
		}
		StateChange enemyDirection = {0.0f, 0.0f};

		float enemyPlayerDistance = std::hypot(playerState.x - enemyState.x, playerState.y - enemyState.y);
		enemyDirection.x = (playerState.x - enemyState.x)/enemyPlayerDistance;
		enemyDirection.y = (playerState.y - enemyState.y)/enemyPlayerDistance;
		enemyState.x += enemyDirection.x * kEnemySpeed * dt;
		enemyState.y += enemyDirection.y * kEnemySpeed * dt;

		enemyHitBox.x = (int)enemyState.x;
		enemyHitBox.y = (int)enemyState.y;

		StateChange playerDirection = {0.0f, 0.0f}; 

		const Uint8* currentKeyStates = SDL_GetKeyboardState(NULL);
		if (currentKeyStates[SDL_SCANCODE_W]) { playerDirection.y-= 1.0f; }
		if (currentKeyStates[SDL_SCANCODE_S]) { playerDirection.y+= 1.0f; }
		if (currentKeyStates[SDL_SCANCODE_A]) { playerDirection.x-= 1.0f; }
		if (currentKeyStates[SDL_SCANCODE_D]) { playerDirection.x+= 1.0f; }

		float playerDirectionMagnitude = std::hypot(playerDirection.x, playerDirection.y);

		if (playerDirectionMagnitude > 1.0f){
			playerDirection.x /= playerDirectionMagnitude;
			playerDirection.y /= playerDirectionMagnitude;
		}

		playerState.x += playerDirection.x * kPlayerSpeed * dt;
		playerState.y += playerDirection.y * kPlayerSpeed * dt;

		playerHitBox.x = (int)playerState.x;
		playerHitBox.y = (int)playerState.y;


		
		previous_time = current_time;
		// Init renderer
		SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF);
		SDL_RenderClear(renderer);
		// Set player color	
		// SDL_SetRenderDrawColor(renderer, 0xFF, 0x00, 0x00, 0x00);
		// SDL_RenderFillRect(renderer, &playerHitBox);
		// Render texture 
		SDL_RenderCopy(renderer, playerTexture, NULL, &playerHitBox);
		SDL_RenderCopy(renderer, enemyTexture, NULL, &enemyHitBox);
		SDL_RenderPresent(renderer);
	}

	SDL_DestroyTexture(playerTexture);
	IMG_Quit();
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();


	return 0;
}
