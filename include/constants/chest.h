// include/constants/chest.h
#ifndef RL2_CONSTANTS_CHEST_H_
#define RL2_CONSTANTS_CHEST_H_

namespace arelto {

constexpr int kChestSpriteSheetCols = 4;
constexpr int kChestSpriteSheetRows = 3;
constexpr int kChestNumSpriteCells =
    kChestSpriteSheetCols * kChestSpriteSheetRows;
constexpr int kChestSpriteCellWidth = 275;
constexpr int kChestSpriteCellHeight = 200;

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
