// include/constants/render.h
#ifndef RL2_CONSTANTS_RENDER_H_
#define RL2_CONSTANTS_RENDER_H_

#include <SDL_pixels.h>
namespace rl2 {
// Render constants
constexpr float kTexCoordTop = 0.0f;
constexpr float kTexCoordBottom = 1.0f;
constexpr float kTexCoordLeft = 0.0f;
constexpr float kTexCoordRight = 1.0f;
constexpr int kSpriteColliderMargin = 30;
constexpr float kRenderCullPadding = 50.0f;

// Colors
// Format is { R, G, B, A }, where A is opacity (alpha)
constexpr SDL_Color kColorWhite = {255, 255, 255, 255};
constexpr SDL_Color kColorGrey = {140, 140, 140, 255};
constexpr SDL_Color kColorGreen = {0, 255, 0, 255};
constexpr SDL_Color kColorBlue = {0, 0, 255, 255};
constexpr SDL_Color kColorRed = {255, 0, 0, 255};
constexpr SDL_Color kColorYellow = {255, 255, 0, 255};
constexpr SDL_Color kColorOrange = {255, 165, 0, 255};
constexpr SDL_Color kColorBlack = {0, 0, 0, 255};

constexpr SDL_Color WithOpacity(SDL_Color color, Uint8 opacity) {
  return {color.r, color.g, color.b, opacity};
}

}  // namespace rl2
#endif
