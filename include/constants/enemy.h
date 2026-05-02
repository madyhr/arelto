// include/constants/enemy.h
#ifndef RL2_CONSTANTS_ENEMY_H_
#define RL2_CONSTANTS_ENEMY_H_
#include <cstddef>

namespace arelto {

constexpr int kNumEnemies = 100;
// Num frames in the animation sprite sheet
constexpr int kEnemyNumSpriteCells = 9;
constexpr int kEnemyAnimationFrameDuration = 150;  // time in ms
constexpr int kEnemyMinimumInitialDistance = 300;
constexpr size_t kEnemyOccupancyMapWidth = 20;
constexpr size_t kEnemyOccupancyMapHeight = 20;
constexpr int kEnemyVertices = 6;
constexpr int kTotalEnemyVertices = kEnemyVertices * kNumEnemies;
}  // namespace arelto
#endif
