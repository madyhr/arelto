#include <SDL2/SDL_error.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_keyboard.h>
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_image.h>
#include <iostream>
#include <SDL2/SDL.h>


const float kBoxInitX = 400.0f;
const float kBoxInitY = 300.0f;

const float kBoxSpeed = 200.0f;

const int kBoxHeight = 75;
const int kBoxWidth = 75;

const int kWindowHeight = 600;
const int kWindowWidth = 800;

struct Position {
	float x;
	float y;
};

Position boxState = {kBoxInitX, kBoxInitY};

struct StateChange {
	float x;
	float y;
};

StateChange boxStateChange = {0.0f,0.0f};

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

	SDL_Texture* boxTexture = IMG_LoadTexture(renderer, "wizard.png");

	if (boxTexture == NULL) {
		std::cerr << "Texture could not be loaded: " << SDL_GetError() << std::endl;
		SDL_DestroyRenderer(renderer);
		SDL_DestroyWindow(window);
		IMG_Quit();
		SDL_Quit();
		return 1;
	}	



	SDL_Rect boxRect = {
		(int)boxState.x, // x pos
		(int)boxState.y, // y pos
		kBoxWidth, // width
		kBoxHeight// height
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


		StateChange direction = {0.0f, 0.0f}; 
		const Uint8* currentKeyStates = SDL_GetKeyboardState(NULL);
		if (currentKeyStates[SDL_SCANCODE_W]) { direction.y-= 1.0f; }
		if (currentKeyStates[SDL_SCANCODE_S]) { direction.y+= 1.0f; }
		if (currentKeyStates[SDL_SCANCODE_A]) { direction.x-= 1.0f; }
		if (currentKeyStates[SDL_SCANCODE_D]) { direction.x+= 1.0f; }

		float magnitude = std::sqrt(
			direction.x * direction.x + direction.y * direction.y
		);

		if (magnitude > 1.0f){
			direction.x /= magnitude;
			direction.y /= magnitude;
		}

		boxState.x += direction.x * kBoxSpeed * dt;
		boxState.y += direction.y * kBoxSpeed * dt;

		boxRect.x = (int)boxState.x;
		boxRect.y = (int)boxState.y;

		previous_time = current_time;
		boxStateChange.x = 0.0f;
		boxStateChange.y = 0.0f;
		// Init renderer
		SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF);
		SDL_RenderClear(renderer);
		// Set box color	
		// SDL_SetRenderDrawColor(renderer, 0xFF, 0x00, 0x00, 0x00);
		// SDL_RenderFillRect(renderer, &boxRect);
		// Render texture 
		SDL_RenderCopy(renderer, boxTexture, NULL, &boxRect);
		SDL_RenderPresent(renderer);


	}

	SDL_DestroyTexture(boxTexture);
	IMG_Quit();
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();


	return 0;
}
