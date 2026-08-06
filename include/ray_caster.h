// include/ray_caster.h
#ifndef RL2_RAY_CASTER_H_
#define RL2_RAY_CASTER_H_

#include <array>
#include <vector>
#include "constants/enemy.h"
#include "constants/ray_caster.h"
#include "map.h"
#include "types.h"
namespace arelto {

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

struct DualRayHit {
  RayHit blocking_hit;
  RayHit non_blocking_hit;
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

  RayHistoryDistances non_blocking_ray_hit_distances = {};
  RayHistoryTypes non_blocking_ray_hit_types = {};

  // the current head of the history buffer
  int history_idx = 0;

  void Reset() {
    ray_hit_distances.fill({});
    ray_hit_types.fill({});
    non_blocking_ray_hit_distances.fill({});
    non_blocking_ray_hit_types.fill({});
    history_idx = 0;
  }

  void ResetEnemy(int enemy_idx) {
    for (int history_idx = 0; history_idx < kRayHistoryLength; history_idx++) {
      for (int ray = 0; ray < kNumRays; ray++) {

        ray_hit_distances[history_idx][ray][enemy_idx] = 0.0f;
        ray_hit_types[history_idx][ray][enemy_idx] = EntityType::None;
        non_blocking_ray_hit_distances[history_idx][ray][enemy_idx] = 0.0f;
        non_blocking_ray_hit_types[history_idx][ray][enemy_idx] =
            EntityType::None;
      }
    }
  }
};

struct Ray {
  Vector2D start_pos;
  Vector2D ray_dir;
};

struct RayHistoryIndex {
  size_t history_idx;
  size_t enemy_idx;
};

void SetupEnemyRayCasterPattern(EnemyRayCaster& pattern);
DualRayHit CastRay(
    const Ray& ray,
    const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map);
bool IsEntityTypePresent(const RayHistoryTypes& ray_hit_types,
                         RayHistoryIndex index, EntityType target);

}  // namespace arelto

#endif
