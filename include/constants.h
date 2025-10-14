// include/constants.h
#ifndef RL2_CONSTANTS_H_
#define RL2_CONSTANTS_H_

namespace rl2 {
// Window constants
const int kWindowHeight = 720;
const int kWindowWidth = 1280;

// Player constants
const float kPlayerInitX = 400.0f;
const float kPlayerInitY = 300.0f;
const float kPlayerSpeed = 200.0f;
const int kPlayerHeight = 100;
const int kPlayerWidth = 100;

// Enemy constants
const int kNumEnemies = 10;
const int kEnemyHealth = 1;
const float kEnemyInitX = 100.0f;
const float kEnemyInitY = 100.0f;
const float kEnemySpeed = 40.0f;
const int kEnemyHeight = 75;
const int kEnemyWidth = 75;

} // namespace rl2
#endif
