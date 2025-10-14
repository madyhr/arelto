// include/constexprants.h
#ifndef RL2_constexprANTS_H_
#define RL2_constexprANTS_H_

namespace rl2 {
// Window constexprants
constexpr int kWindowHeight = 720;
constexpr int kWindowWidth = 1280;

// Player constexprants
constexpr float kPlayerInitX = 400.0f;
constexpr float kPlayerInitY = 300.0f;
constexpr float kPlayerSpeed = 200.0f;
constexpr int kPlayerHeight = 100;
constexpr int kPlayerWidth = 100;

// Enemy constexprants
constexpr int kNumEnemies = 10;
constexpr int kEnemyHealth = 1;
constexpr float kEnemyInitX = 100.0f;
constexpr float kEnemyInitY = 100.0f;
constexpr float kEnemySpeed = 40.0f;
constexpr int kEnemyHeight = 75;
constexpr int kEnemyWidth = 75;

constexpr int kEnemyVertices = 6;
constexpr int kTotalEnemyVertices = kEnemyVertices * kNumEnemies;
constexpr float kTexCoordTop = 0.0f;
constexpr float kTexCoordBottom = 1.0f;
constexpr float kTexCoordLeft = 0.0f;
constexpr float kTexCoordRight = 1.0f;

}  // namespace rl2
#endif
