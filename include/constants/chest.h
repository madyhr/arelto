// include/constants/chest.h
#ifndef RL2_CONSTANTS_CHEST_H_
#define RL2_CONSTANTS_CHEST_H_

#include "render.h"
#include "types.h"

namespace arelto {

constexpr float kChestSpawnChance = 0.99f;

constexpr int kChestSpriteSheetCols = 4;
constexpr int kChestSpriteSheetRows = 3;
constexpr int kChestNumSpriteCells =
    kChestSpriteSheetCols * kChestSpriteSheetRows;
constexpr int kChestSpriteCellWidth = 275;
constexpr int kChestSpriteCellHeight = 200;

constexpr int kChestSpriteWidth = 70;
constexpr int kChestSpriteHeight = 55;
constexpr int kChestColliderWidth =
    kChestSpriteWidth - kSpriteColliderMargin / 2;
constexpr int kChestColliderHeight =
    kChestSpriteHeight - kSpriteColliderMargin / 2;

constexpr Size2D kChestSpriteSize = {kChestSpriteWidth, kChestSpriteHeight};
constexpr Collider kChestCollider = {
    {0.5f * kChestSpriteWidth, 0.5f * kChestSpriteHeight},
    {kChestColliderWidth, kChestColliderHeight}};

constexpr int kChestVertices = 6;

// Constants for the chest opening animation.
constexpr int kChestAnimationWidth = 550;
constexpr int kChestAnimationHeight = 400;
constexpr int kChestAnimationFrameDuration = 100;  // ms per frame
constexpr int kChestLoopCount = 2;
// Total animation frames: 12 (total number of cells) + 8*loop_count (last 2 rows each with 4 columns loops)
constexpr int kChestTotalAnimFrames =
    kChestNumSpriteCells + (kChestSpriteSheetCols * 2) * kChestLoopCount;

}  // namespace arelto
#endif
