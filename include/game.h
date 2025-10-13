// include/game.h
#ifndef RL2_GAME_H_
#define RL2_GAME_H_

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <cstdint>
#include "entity.h"
#include "constants.h"

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

struct GameEntities {
	Entity* player = nullptr;
	Entity* enemy = nullptr;
};

class Game{
public:
	Game();
	~Game();
	bool Initialize();
	void RunGameLoop();
	void Shutdown();
private:
	GameResources resources;
	Entity player_;
	Entity enemy_;
	void ProcessInput();
	void UpdateGame();
	void GenerateOutput();

	bool is_running_;
	uint32_t ticks_count_ = 0;
};

} // namespace rl2
#endif
