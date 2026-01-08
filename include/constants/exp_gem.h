// include/constants/exp_gem.h
#ifndef RL2_CONSTANTS_EXP_GEM_H_
#define RL2_CONSTANTS_EXP_GEM_H_

#include "constants/render.h"
#include "types.h"

namespace rl2 {
constexpr std::array<int, ExpGemType::Count> kExpGemValues = {1, 2, 4, 8};
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

constexpr int kExpGemMediumSpriteWidth = 30;
constexpr int kExpGemMediumSpriteHeight = 40;
constexpr int kExpGemMediumColliderWidth =
    kExpGemMediumSpriteWidth - kSpriteColliderMargin / 2;
constexpr int kExpGemMediumColliderHeight =
    kExpGemMediumSpriteHeight - kSpriteColliderMargin / 2;

constexpr Size2D kExpGemMediumSpriteSize = {kExpGemMediumSpriteWidth,
                                            kExpGemMediumSpriteHeight};

constexpr Collider kExpGemMediumCollider = {
    {0.5 * kExpGemMediumSpriteWidth, 0.5 * kExpGemMediumSpriteHeight},
    {kExpGemMediumColliderWidth, kExpGemMediumColliderHeight}};

constexpr int kExpGemLargeSpriteWidth = 40;
constexpr int kExpGemLargeSpriteHeight = 53;
constexpr int kExpGemLargeColliderWidth =
    kExpGemLargeSpriteWidth - kSpriteColliderMargin / 2;
constexpr int kExpGemLargeColliderHeight =
    kExpGemLargeSpriteHeight - kSpriteColliderMargin / 2;

constexpr Size2D kExpGemLargeSpriteSize = {kExpGemLargeSpriteWidth,
                                           kExpGemLargeSpriteHeight};

constexpr Collider kExpGemLargeCollider = {
    {0.5 * kExpGemLargeSpriteWidth, 0.5 * kExpGemLargeSpriteHeight},
    {kExpGemLargeColliderWidth, kExpGemLargeColliderHeight}};

constexpr int kExpGemHugeSpriteWidth = 50;
constexpr int kExpGemHugeSpriteHeight = 65;
constexpr int kExpGemHugeColliderWidth =
    kExpGemHugeSpriteWidth - kSpriteColliderMargin / 2;
constexpr int kExpGemHugeColliderHeight =
    kExpGemHugeSpriteHeight - kSpriteColliderMargin / 2;

constexpr Size2D kExpGemHugeSpriteSize = {kExpGemHugeSpriteWidth,
                                          kExpGemHugeSpriteHeight};

constexpr Collider kExpGemHugeCollider = {
    {0.5 * kExpGemHugeSpriteWidth, 0.5 * kExpGemHugeSpriteHeight},
    {kExpGemHugeColliderWidth, kExpGemHugeColliderHeight}};

}  // namespace rl2
#endif
