// include/constants.h
#ifndef RL2_constants_H_
#define RL2_constants_H_
#include <cmath>

namespace rl2 {
// Window constants
constexpr int kWindowWidth = 1920;
constexpr int kWindowHeight = 1080;

// Game constats
constexpr float kPhysicsDt = 0.001f;      // time in sec
constexpr float kEpisodeTimeout = 60.0f;  // time in sec

// UI constants
constexpr float kHealthBarGroupX = 50.0f;
constexpr float kHealthBarGroupY = 900.0f;
constexpr int kHealthBarContainerSpriteOffsetX = 0;
constexpr int kHealthBarContainerSpriteOffsetY = 0;
constexpr int kHealthBarContainerSpriteWidth = 404;
constexpr int kHealthBarContainerSpriteHeight = 92;
constexpr float kHealthBarContainerRelOffsetX = 0.0f;
constexpr float kHealthBarContainerRelOffsetY = 0.0f;
constexpr float kHealthBarRelOffsetX = 80.0f;
constexpr float kHealthBarRelOffsetY = 32.0f;
constexpr int kHealthBarSpriteOffsetX = 0;
constexpr int kHealthBarSpriteOffsetY = 128;
constexpr int kHealthBarSpriteWidth = 299;
constexpr int kHealthBarSpriteHeight = 28;
constexpr int kHealthBarTextRelOffsetX = 100;
constexpr int kHealthBarTextRelOffsetY = 32;
constexpr int kDigitSpriteWidth = 30;
constexpr int kDigitSpriteHeight = 50;
constexpr int kHealthBarTextCharWidth = 20;
constexpr int kHealthBarTextCharHeight = 25;
constexpr float kTimerGroupX = 50.0f;
constexpr float kTimerGroupY = 50.0f;
constexpr int kTimerHourglassSpriteWidth = 50;
constexpr int kTimerHourglassSpriteHeight = 72;
constexpr int kTimerHourglassRelOffsetX = 0;
constexpr int kTimerHourglassRelOffsetY = 0;
constexpr int kTimerTextRelOffsetX = 60;
constexpr int kTimerTextRelOffsetY = 0;
constexpr int kTimerTextCharWidth = 50;
constexpr int kTimerTextCharHeight = 72;
constexpr int kGameOverSpriteWidth = 610;
constexpr int kGameOverSpriteHeight = 88;
constexpr int kPausedSpriteWidth = 610;
constexpr int kPausedSpriteHeight = 120;

// Map constants
constexpr int kMapWidth = 10000;
constexpr int kMapHeight = 10000;
// the inverse map max distance is used for scaling distances.
const float kInvMapMaxDistance =
    1.0f / std::sqrt(static_cast<float>(kMapHeight) * kMapHeight +
                     static_cast<float>(kMapWidth) * kMapWidth);
constexpr int kOccupancyMapResolution = 25;
constexpr int kOccupancyMapWidth =
    (kMapWidth + kOccupancyMapResolution - 1) / kOccupancyMapResolution;
constexpr int kOccupancyMapHeight =
    (kMapHeight + kOccupancyMapResolution - 1) / kOccupancyMapResolution;
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
constexpr int kSpriteColliderMargin = 30;
constexpr float kRenderCullPadding = 50.0f;

// Game status constants
// number of frames to average over in fps counter
constexpr int kFrameTimes = 1000;
constexpr float kMaxFrameTime = 0.1;  // in seconds

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

// Abilities constants
constexpr int kNumPlayerSpells = 2;  // total number of spells

// Observation constants
constexpr float kPositionObservationScale = 1000.0f;

// Raycaster constants
constexpr int kRayHistoryLength = 4;
constexpr int kNumRays = 72;
constexpr float kMaxRayDistance = 5000.0f;
constexpr float kMinRayDistance = 30.0f;  // offset from start in dir of ray

}  // namespace rl2
#endif
