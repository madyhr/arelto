// include/constants.h
#ifndef RL2_constants_H_
#define RL2_constants_H_
#include <cmath>

namespace rl2 {
// Window constants
constexpr int kWindowWidth = 1920;
constexpr int kWindowHeight = 1080;

// Map constants
constexpr int kMapWidth = 3000;
constexpr int kMapHeight = 3000;
constexpr int kTileSize = 256;
constexpr int kNumTileTypes = 16;
constexpr int kNumTilesX = ((kMapWidth + kTileSize - 1) / kTileSize);
constexpr int kNumTilesY = ((kMapHeight + kTileSize - 1) / kTileSize);

// Game status constants
constexpr int kFrameTimes = 1000;

// Player constants
constexpr float kPlayerInitX = 500.0f;
constexpr float kPlayerInitY = 500.0f;
constexpr float kPlayerSpeed = 200.0f;
constexpr int kPlayerWidth = 60;
// Derived from the generated wizard png
constexpr int kPlayerHeight = (int)(kPlayerWidth * 1.258);
constexpr float kPlayerInvMass = 0.01f;

// Enemy constants
constexpr int kNumEnemies = 10;
constexpr int kEnemyHealth = 5;
constexpr float kEnemyInitX = 100.0f;
constexpr float kEnemyInitY = 100.0f;
constexpr float kEnemySpeed = 40.0f;
constexpr int kEnemyWidth = 30;
// Derived from the generated goblin png
constexpr int kEnemyHeight = (int)(kEnemyWidth * 1.04);
constexpr float kEnemyInvMass = 0.1f;

constexpr int kEnemyMinimumInitialDistance = 500;
constexpr int kEnemyVertices = 6;
constexpr int kTotalEnemyVertices = kEnemyVertices * kNumEnemies;
constexpr float kTexCoordTop = 0.0f;
constexpr float kTexCoordBottom = 1.0f;
constexpr float kTexCoordLeft = 0.0f;
constexpr float kTexCoordRight = 1.0f;

// Projectiles constants
constexpr int kProjectileVertices = 6;
constexpr int kFireballWidth = 50;
constexpr int kFireballHeight = 50;
constexpr float kFireballSpeed = 350.0f;
constexpr int kFireBallDamage = 5;
constexpr int kFrostboltWidth = 100;
constexpr int kFrostboltHeight = 100;
constexpr float kFrostboltSpeed = 250.0f;
constexpr int kFrostboltDamage = 10;

// Abilities constants
constexpr int kNumSpells = 2; // total number of spells
constexpr int kNumProjectileSpells = 2;

// Entity constants
constexpr int kNumEntities = kNumEnemies + 1;
}  // namespace rl2
#endif
