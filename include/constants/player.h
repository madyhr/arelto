// include/constants/player.h
#ifndef RL2_CONSTANTS_PLAYER_H_
#define RL2_CONSTANTS_PLAYER_H_
#include "constants/map.h"
#include "constants/render.h"

namespace rl2 {
// Player constants
constexpr int kPlayerInitMaxHealth = 100;
constexpr float kPlayerInitX = kMapWidth * 0.5f;
constexpr float kPlayerInitY = kMapHeight * 0.5f;
constexpr float kPlayerSpeed = 200.0f;
constexpr int kPlayerSpriteWidth = 60;
// Derived from the generated wizard png
constexpr int kPlayerSpriteHeight = (int)(kPlayerSpriteWidth * 1.258);
constexpr float kPlayerColliderOffsetX = 0.5f * kPlayerSpriteWidth;
constexpr float kPlayerColliderOffsetY = 0.5f * kPlayerSpriteHeight;
constexpr int kPlayerColliderWidth = kPlayerSpriteWidth - kSpriteColliderMargin;
constexpr int kPlayerColliderHeight =
    kPlayerSpriteHeight - kSpriteColliderMargin;
constexpr float kPlayerInvMass = 0.01f;
// Num frames in the animation sprite sheet
constexpr int kPlayerNumSpriteCells = 9;
constexpr int kPlayerSpriteCellWidth = 48;
constexpr int kPlayerSpriteCellHeight = 64;
constexpr int kPlayerAnimationFrameDuration = 150;  // time in ms
constexpr float kPlayerInvulnerableWindow = 0.1;    // time in sec
                                                    //
// Abilities constants
constexpr int kNumPlayerSpells = 2;  // total number of spells

}  // namespace rl2
#endif
