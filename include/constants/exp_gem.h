// include/constants/exp_gem.h
#ifndef RL2_CONSTANTS_EXP_GEM_H_
#define RL2_CONSTANTS_EXP_GEM_H_

#include "constants/render.h"
#include "types.h"

namespace arelto {
constexpr int kExpGemVertices = 6;

constexpr int kExpGemCommonSpriteWidth = 25;
constexpr int kExpGemCommonSpriteHeight = 33;
constexpr int kExpGemCommonColliderWidth =
    kExpGemCommonSpriteWidth - kSpriteColliderMargin / 2;
constexpr int kExpGemCommonColliderHeight =
    kExpGemCommonSpriteHeight - kSpriteColliderMargin / 2;

constexpr Size2D kExpGemCommonSpriteSize = {kExpGemCommonSpriteWidth,
                                            kExpGemCommonSpriteHeight};

constexpr Collider kExpGemCommonCollider = {
    {0.5 * kExpGemCommonSpriteWidth, 0.5 * kExpGemCommonSpriteHeight},
    {kExpGemCommonColliderWidth, kExpGemCommonColliderHeight}};

constexpr int kExpGemRareSpriteWidth = 30;
constexpr int kExpGemRareSpriteHeight = 40;
constexpr int kExpGemRareColliderWidth =
    kExpGemRareSpriteWidth - kSpriteColliderMargin / 2;
constexpr int kExpGemRareColliderHeight =
    kExpGemRareSpriteHeight - kSpriteColliderMargin / 2;

constexpr Size2D kExpGemRareSpriteSize = {kExpGemRareSpriteWidth,
                                          kExpGemRareSpriteHeight};

constexpr Collider kExpGemRareCollider = {
    {0.5 * kExpGemRareSpriteWidth, 0.5 * kExpGemRareSpriteHeight},
    {kExpGemRareColliderWidth, kExpGemRareColliderHeight}};

constexpr int kExpGemEpicSpriteWidth = 35;
constexpr int kExpGemEpicSpriteHeight = 45;
constexpr int kExpGemEpicColliderWidth =
    kExpGemEpicSpriteWidth - kSpriteColliderMargin / 2;
constexpr int kExpGemEpicColliderHeight =
    kExpGemEpicSpriteHeight - kSpriteColliderMargin / 2;

constexpr Size2D kExpGemEpicSpriteSize = {kExpGemEpicSpriteWidth,
                                          kExpGemEpicSpriteHeight};

constexpr Collider kExpGemEpicCollider = {
    {0.5 * kExpGemEpicSpriteWidth, 0.5 * kExpGemEpicSpriteHeight},
    {kExpGemEpicColliderWidth, kExpGemEpicColliderHeight}};

constexpr int kExpGemLegendarySpriteWidth = 45;
constexpr int kExpGemLegendarySpriteHeight = 60;
constexpr int kExpGemLegendaryColliderWidth =
    kExpGemLegendarySpriteWidth - kSpriteColliderMargin / 2;
constexpr int kExpGemLegendaryColliderHeight =
    kExpGemLegendarySpriteHeight - kSpriteColliderMargin / 2;

constexpr Size2D kExpGemLegendarySpriteSize = {kExpGemLegendarySpriteWidth,
                                               kExpGemLegendarySpriteHeight};

constexpr Collider kExpGemLegendaryCollider = {
    {0.5 * kExpGemLegendarySpriteWidth, 0.5 * kExpGemLegendarySpriteHeight},
    {kExpGemLegendaryColliderWidth, kExpGemLegendaryColliderHeight}};

constexpr std::array<int, Rarity::Count> kExpGemValues = {1, 2, 4, 8};
constexpr std::array<Size2D, Rarity::Count> kExpGemSpriteSize = {
    kExpGemCommonSpriteSize, kExpGemRareSpriteSize, kExpGemEpicSpriteSize,
    kExpGemLegendarySpriteSize};

constexpr std::array<Collider, Rarity::Count> kExpGemCollider = {
    kExpGemCommonCollider, kExpGemRareCollider, kExpGemEpicCollider,
    kExpGemLegendaryCollider};

}  // namespace arelto
#endif
