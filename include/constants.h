// include/constexprants.h
#ifndef RL2_constexprANTS_H_
#define RL2_constexprANTS_H_

namespace rl2 {
// Window constants
constexpr int kWindowWidth = 1920;
constexpr int kWindowHeight = 1080;

// Map constants
constexpr int kMapWidth = 10000;
constexpr int kMapHeight = 10000;

// Game status constants
constexpr int kFrameTimes = 1000;

// Player constants
constexpr float kPlayerInitX = 960.0f;
constexpr float kPlayerInitY = 540.0f;
constexpr float kPlayerSpeed = 200.0f;
constexpr int kPlayerWidth = 60;
// Derived from the generated wizard png
constexpr int kPlayerHeight = (int)(kPlayerWidth*1.258);
constexpr float kPlayerInvMass = 0.01f;

// Enemy constants
constexpr int kNumEnemies = 100;
constexpr int kEnemyHealth = 1;
constexpr float kEnemyInitX = 100.0f;
constexpr float kEnemyInitY = 100.0f;
constexpr float kEnemySpeed = 40.0f;
constexpr int kEnemyWidth = 30;
// Derived from the generated goblin png
constexpr int kEnemyHeight = (int)(kEnemyWidth*1.04);
constexpr float kEnemyInvMass = 0.1f;

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
