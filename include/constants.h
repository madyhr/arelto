// include/constants.h
#ifndef RL2_constants_H_
#define RL2_constants_H_
#include <cmath>

namespace rl2 {
// Window constants
constexpr int kWindowWidth = 1920;
constexpr int kWindowHeight = 1080;
constexpr float kCullPadding = 50.0f;

// Game constats
constexpr float kPhysicsDt = 0.001f;  // time in sec

// Map constants
constexpr int kMapWidth = 5000;
constexpr int kMapHeight = 5000;
constexpr int kOccupancyMapResolution = 25;
constexpr int kOccupancyMapWidth =
    static_cast<int>(kMapWidth / kOccupancyMapResolution);
constexpr int kOccupancyMapHeight =
    static_cast<int>(kMapHeight / kOccupancyMapResolution);
// number of physics steps per occupancy map update
constexpr int kOccupancyMapTimeDecimation = 100;
// Tiles are used for rendering
constexpr int kTileWidth = 40;
constexpr int kTileHeight = 119;
constexpr int kNumTileTypes = 44;
constexpr int kNumTilesX = ((kMapWidth + kTileWidth - 1) / kTileWidth);
constexpr int kNumTilesY = ((kMapHeight + kTileHeight - 1) / kTileHeight);

// Render constants
constexpr float kTexCoordTop = 0.0f;
constexpr float kTexCoordBottom = 1.0f;
constexpr float kTexCoordLeft = 0.0f;
constexpr float kTexCoordRight = 1.0f;
constexpr int kSpriteColliderMargin = 7;

// Game status constants
// number of frames to average over in fps counter
constexpr int kFrameTimes = 1000;
constexpr float kMaxFrameTime = 0.1;  // in seconds

// Player constants
constexpr float kPlayerInitX = 500.0f;
constexpr float kPlayerInitY = 500.0f;
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

// Enemy constants
constexpr int kNumEnemies = 10;
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

// Projectiles constants
constexpr int kProjectileVertices = 6;
constexpr int kProjectileNumSpriteCells = 12;
constexpr int kProjectileSpriteCellWidth = 24;
constexpr int kProjectileSpriteCellHeight = 48;
constexpr int kProjectileAnimationFrameDuration = 150;  // time in ms
constexpr int kFireballSpriteWidth = 40;
constexpr int kFireballSpriteHeight = 40;
constexpr int kFireballColliderWidth =
    kFireballSpriteWidth - kSpriteColliderMargin;
constexpr int kFireballColliderHeight =
    kFireballSpriteHeight - kSpriteColliderMargin;
constexpr float kFireballSpeed = 350.0f;
constexpr int kFireBallDamage = 5;
constexpr float kFireballCooldown = 1.0f;  // time in sec
constexpr int kFrostboltSpriteWidth = 100;
constexpr int kFrostboltSpriteHeight = 100;
constexpr int kFrostboltColliderWidth =
    kEnemySpriteWidth - kSpriteColliderMargin;
constexpr int kFrostboltColliderHeight =
    kEnemySpriteHeight - kSpriteColliderMargin;
constexpr float kFrostboltSpeed = 250.0f;
constexpr int kFrostboltDamage = 10;
constexpr float kFrostboltCooldown = 2.0f;  // time in sec

// Abilities constants
constexpr int kNumPlayerSpells = 2;  // total number of spells

}  // namespace rl2
#endif
