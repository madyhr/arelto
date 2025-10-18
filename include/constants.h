// include/constexprants.h
#ifndef RL2_constexprANTS_H_
#define RL2_constexprANTS_H_

namespace rl2 {
// Window constants
constexpr int kWindowHeight = 1080;
constexpr int kWindowWidth = 1920;

// Player constants
constexpr float kPlayerInitX = 960.0f;
constexpr float kPlayerInitY = 540.0f;
constexpr float kPlayerSpeed = 200.0f;
constexpr int kPlayerWidth = 100;
constexpr int kPlayerHeight = (int)(kPlayerWidth*1.258);

// Enemy constants
constexpr int kNumEnemies = 20;
constexpr int kEnemyHealth = 1;
constexpr float kEnemyInitX = 100.0f;
constexpr float kEnemyInitY = 100.0f;
constexpr float kEnemySpeed = 40.0f;
constexpr int kEnemyWidth = 75;
constexpr int kEnemyHeight = (int)(kEnemyWidth*1.04);

constexpr int kEnemyMinimumInitialDistance = 500;
constexpr int kEnemyVertices = 6;
constexpr int kTotalEnemyVertices = kEnemyVertices * kNumEnemies;
constexpr float kTexCoordTop = 0.0f;
constexpr float kTexCoordBottom = 1.0f;
constexpr float kTexCoordLeft = 0.0f;
constexpr float kTexCoordRight = 1.0f;

// Entity constants
constexpr int kNumEntities = kNumEnemies + 1;
}  // namespace rl2
#endif
