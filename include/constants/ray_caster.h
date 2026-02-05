// include/constants/ray_caster.h
#ifndef RL2_CONSTANTS_RAY_CASTER_H_
#define RL2_CONSTANTS_RAY_CASTER_H_

namespace arelto {
// Observation constants


// Raycaster constants
constexpr int kRayHistoryLength = 4;
constexpr int kNumRays = 72;
constexpr float kMaxRayDistance = 5000.0f;
constexpr float kMinRayDistance = 30.0f;  // offset from start in dir of ray

}  // namespace arelto
#endif
