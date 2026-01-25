// include/constants/game.h
#ifndef RL2_CONSTANTS_GAME_H_
#define RL2_CONSTANTS_GAME_H_

namespace rl2 {
// Game constats
constexpr float kPhysicsDt = 0.001f;      // time in sec
constexpr float kEpisodeTimeout = 60.0f;  // time in sec

// Game status constants
// number of frames to average over in fps counter
constexpr int kFrameTimes = 1000;
constexpr float kMaxFrameTime = 0.1;  // in seconds
constexpr int kWindowWidth = 1920;
constexpr int kWindowHeight = 1080;

}  // namespace rl2
#endif
