#include <SDL2/SDL_error.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_keyboard.h>
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_image.h>
#include <iostream>
#include <SDL2/SDL.h>


const float BOX_INIT_X = 400.0f;
const float BOX_INIT_Y = 300.0f;

const float BOX_VELOCITY = 1.0f;

const int BOX_HEIGHT = 50;
const int BOX_WIDTH = 50;

const int SCREEN_SIZE_HEIGHT = 600;
const int SCREEN_SIZE_WIDTH = 800;

struct Position {
	float x;
	float y;
};

Position boxState = {BOX_INIT_X, BOX_INIT_Y};


int main(int argc, char* args[]){

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::cerr << "SDL could not initialize! SDL Error: " << SDL_GetError() << std::endl;
		return 1;
	}
	

	SDL_Window* window = SDL_CreateWindow(
		"My_CPP_Game",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		SCREEN_SIZE_WIDTH,
		SCREEN_SIZE_HEIGHT,
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


	int box_width_local;
	int box_height_local;

	SDL_QueryTexture(boxTexture, NULL, NULL, &box_width_local, &box_height_local);

	SDL_Rect boxRect = {
		(int)boxState.x, // x pos
		(int)boxState.y, // y pos
		box_width_local, // width
		box_height_local// height
	};
	
	bool quit = false;
	SDL_Event e;

	while (!quit){
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


		const Uint8* currentKeyStates = SDL_GetKeyboardState(NULL);
		if (currentKeyStates[SDL_SCANCODE_W]) { boxState.y-= BOX_VELOCITY; }
		if (currentKeyStates[SDL_SCANCODE_S]) { boxState.y+= BOX_VELOCITY; }
		if (currentKeyStates[SDL_SCANCODE_A]) { boxState.x-= BOX_VELOCITY; }
		if (currentKeyStates[SDL_SCANCODE_D]) { boxState.x+= BOX_VELOCITY; }

		boxRect.x = (int)boxState.x;
		boxRect.y = (int)boxState.y;

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
