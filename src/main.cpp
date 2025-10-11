#include <SDL2/SDL_error.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_video.h>
#include <iostream>
#include <SDL2/SDL.h>

const int SCREEN_SIZE_HEIGHT = 600;
const int SCREEN_SIZE_WIDTH = 800;

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
	
	bool quit = false;
	SDL_Event e;

	while (!quit){
		while (SDL_PollEvent(&e) != 0) {
			if (e.type == SDL_QUIT) {
				quit = true;
			}
			else if (e.type == SDL_KEYDOWN){

				switch (e.key.keysym.sym){
					case SDLK_UP:
						std::cout << "Key UP pressed!" << std::endl;
						break;
					case SDLK_1:
						std::cout << "Key 1 pressed!" << std::endl;
						break;

					case SDLK_w:
						std::cout << "Key w pressed!" << std::endl;
						break;
					case SDLK_q:
						quit = true;
						std::cout << "Key q pressed! EXITING!!!" << std::endl;
						break;
				}
			}
		}
	}

	SDL_DestroyWindow(window);
	SDL_Quit();


	return 0;
}
