// include/ray_caster.h
#ifndef RL2_RAY_CASTER_H_
#define RL2_RAY_CASTER_H_

#include <array>
#include <vector>
#include "constants/enemy.h"
#include "constants/ray_caster.h"
#include "map.h"
#include "types.h"
namespace rl2 {

using RayHistoryDistances =
    std::array<std::array<std::array<float, kNumEnemies>, kNumRays>,
               kRayHistoryLength>;

using RayHistoryTypes =
    std::array<std::array<std::array<EntityType, kNumEnemies>, kNumRays>,
               kRayHistoryLength>;

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
  RayHistoryDistances ray_hit_distances = {};
  RayHistoryTypes ray_hit_types = {};
  // the current head of the history buffer
  int history_idx = 0;
};

void SetupEnemyRayCasterPattern(EnemyRayCaster& pattern);
RayHit CastRay(
    const Vector2D& start_pos, const Vector2D& ray_dir,
    const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map);
bool IsEntityTypePresent(const RayHistoryTypes& ray_hit_types,
                         size_t history_idx, size_t enemy_idx,
                         EntityType target);

}  // namespace rl2

#endif
