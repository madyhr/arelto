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
// Format is { R, G, B, A }, where A is opacity
constexpr SDL_Color kColorWhite = {255, 255, 255, 255};
constexpr SDL_Color kColorGrey = {140, 140, 140, 255};
constexpr SDL_Color kColorGreen = {0, 255, 0, 255};
}  // namespace rl2
#endif
