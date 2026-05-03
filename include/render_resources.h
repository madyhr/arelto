// include/render_resources.h
#ifndef RL2_RENDER_RESOURCES_H_
#define RL2_RENDER_RESOURCES_H_

#include <SDL2/SDL_render.h>
#include <SDL2/SDL_ttf.h>
#include <vector>

namespace arelto {

struct RenderResources {
  // Game textures
  SDL_Texture* tile = nullptr;
  SDL_Texture* player = nullptr;
  SDL_Texture* enemy = nullptr;
  SDL_Texture* chest = nullptr;
  std::vector<SDL_Texture*> projectiles;
  std::vector<SDL_Texture*> gems;
  std::vector<SDL_Texture*> items;

  // UI textures
  SDL_Texture* digit_font = nullptr;
  SDL_Texture* health_bar = nullptr;
  SDL_Texture* exp_bar = nullptr;
  SDL_Texture* level_indicator = nullptr;
  SDL_Texture* timer_hourglass = nullptr;
  SDL_Texture* game_over = nullptr;
  SDL_Texture* start_screen = nullptr;
  SDL_Texture* level_up_option_card = nullptr;
  SDL_Texture* button = nullptr;
  SDL_Texture* begin_button = nullptr;
  SDL_Texture* settings_menu_background = nullptr;
  SDL_Texture* slider = nullptr;
  SDL_Texture* checkbox = nullptr;
  SDL_Texture* checkmark = nullptr;

  // Fonts
  TTF_Font* font_small = nullptr;
  TTF_Font* font_medium = nullptr;
  TTF_Font* font_large = nullptr;
  TTF_Font* font_huge = nullptr;
};

}  // namespace arelto

#endif
