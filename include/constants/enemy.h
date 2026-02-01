// include/constants/enemy.h
#ifndef RL2_CONSTANTS_ENEMY_H_
#define RL2_CONSTANTS_ENEMY_H_
#include <cstddef>
#include "constants/render.h"

namespace arelto {
// Enemy constants
constexpr int kNumEnemies = 100;
constexpr int kEnemyHealth = 10;
constexpr float kEnemyInitX = 100.0f;
constexpr float kEnemyInitY = 100.0f;
constexpr float kEnemySpeed = 40.0f;
constexpr int kEnemySpriteWidth = 42;
// Derived from the generated goblin png
// constexpr int kEnemyHeight = (int)(kEnemyWidth * 1.04);
// Derived from the Tentacle being spritesheet
constexpr int kEnemySpriteHeight = 50;
constexpr float kEnemyColliderOffsetX = 0.5f * kEnemySpriteWidth;
constexpr float kEnemyColliderOffsetY = 0.5f * kEnemySpriteHeight;
constexpr int kEnemyColliderWidth = kEnemySpriteWidth - kSpriteColliderMargin;
constexpr int kEnemyColliderHeight = kEnemySpriteHeight - kSpriteColliderMargin;
constexpr float kEnemyAttackCooldown = 0.1f;  // time in sec

constexpr float kEnemyInvMass = 0.1f;
// Num frames in the animation sprite sheet
constexpr int kEnemyNumSpriteCells = 9;
constexpr int kEnemySpriteCellWidth = 42;
constexpr int kEnemySpriteCellHeight = 50;
constexpr int kEnemyAnimationFrameDuration = 150;  // time in ms
constexpr int kEnemyMinimumInitialDistance = 300;
constexpr size_t kEnemyOccupancyMapWidth = 20;
constexpr size_t kEnemyOccupancyMapHeight = 20;
constexpr int kEnemyVertices = 6;
constexpr int kTotalEnemyVertices = kEnemyVertices * kNumEnemies;
}  // namespace arelto
#endif
