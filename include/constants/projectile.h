// include/constants/projectile.h
#ifndef RL2_CONSTANTS_PROJECTILE_H_
#define RL2_CONSTANTS_PROJECTILE_H_
#include "constants/render.h"

namespace arelto {
// Projectiles constants
constexpr int kProjectileVertices = 6;
constexpr int kProjectileNumSpriteCells = 12;
constexpr int kProjectileSpriteCellWidth = 24;
constexpr int kProjectileSpriteCellHeight = 48;
constexpr int kProjectileAnimationFrameDuration = 150;  // time in ms
constexpr int kFireballSpriteWidth = 60;
constexpr int kFireballSpriteHeight = 60;
constexpr int kFireballColliderWidth =
    kFireballSpriteWidth - kSpriteColliderMargin;
constexpr int kFireballColliderHeight =
    kFireballSpriteHeight - kSpriteColliderMargin;
constexpr float kFireballSpeed = 500.0f;
constexpr int kFireBallDamage = 5;
constexpr float kFireballCooldown = 1.0f;  // time in sec
constexpr int kFrostboltSpriteWidth = 100;
constexpr int kFrostboltSpriteHeight = 100;
constexpr int kFrostboltColliderWidth =
    kFrostboltSpriteWidth - kSpriteColliderMargin;
constexpr int kFrostboltColliderHeight =
    kFrostboltSpriteHeight - kSpriteColliderMargin;
constexpr float kFrostboltSpeed = 250.0f;
constexpr int kFrostboltDamage = 10;
constexpr float kFrostboltCooldown = 2.0f;  // time in sec
}  // namespace arelto
#endif
