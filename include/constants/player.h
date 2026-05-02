// include/constants/player.h
#ifndef RL2_CONSTANTS_PLAYER_H_
#define RL2_CONSTANTS_PLAYER_H_

namespace arelto {
// Player constants
// Num frames in the animation sprite sheet
constexpr int kPlayerNumSpriteCells = 9;
constexpr int kPlayerSpriteCellWidth = 48;
constexpr int kPlayerSpriteCellHeight = 64;
constexpr int kPlayerAnimationFrameDuration = 150;  // time in ms
// Abilities constants
constexpr int kNumPlayerSpells = 2;  // total number of spells
}  // namespace arelto
#endif
