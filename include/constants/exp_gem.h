// include/constants/exp_gem.h
#ifndef RL2_CONSTANTS_EXP_GEM_H_
#define RL2_CONSTANTS_EXP_GEM_H_

#include "constants/render.h"
#include "types.h"

namespace rl2 {
constexpr int kExpGemVertices = 6;

constexpr int kExpGemSmallSpriteWidth = 25;
constexpr int kExpGemSmallSpriteHeight = 33;
constexpr int kExpGemSmallColliderWidth =
    kExpGemSmallSpriteWidth - kSpriteColliderMargin / 2;
constexpr int kExpGemSmallColliderHeight =
    kExpGemSmallSpriteHeight - kSpriteColliderMargin / 2;

constexpr Size2D kExpGemSmallSpriteSize = {kExpGemSmallSpriteWidth,
                                           kExpGemSmallSpriteHeight};

constexpr Collider kExpGemSmallCollider = {
    {0.5 * kExpGemSmallSpriteWidth, 0.5 * kExpGemSmallSpriteHeight},
    {kExpGemSmallColliderWidth, kExpGemSmallColliderHeight}};

}  // namespace rl2
#endif
