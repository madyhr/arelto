// include/ray_caster.h
#ifndef RL2_RAY_CASTER_H_
#define RL2_RAY_CASTER_H_

#include <array>
#include <vector>
#include "constants.h"
#include "map.h"
#include "types.h"
namespace rl2 {

struct RayCaster {
  Vector2D ray_start;
  std::vector<Vector2D> ray_dirs;
  Vector2D ray_unit_step_size;
  Vector2D ray_length_1d;
};

struct RayHit {
  float distance;
  EntityType entity_type;
};

template <int num_rays>
struct RayCasterPattern {
  std::array<Vector2D, num_rays> ray_dir;
};

struct EnemyRayCaster {
  RayCasterPattern<kNumRays> pattern;
  std::array<Vector2D, kNumEnemies> ray_start_pos;
  std::array<std::array<float, kNumEnemies>, kNumRays> ray_hit_distances;
  std::array<std::array<EntityType, kNumEnemies>, kNumRays> ray_hit_types;
};

void SetupEnemyRayCasterPattern(EnemyRayCaster& pattern);
RayHit CastRay(
    const Vector2D& start_pos, const Vector2D& ray_dir,
    const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map);
}  // namespace rl2

#endif
