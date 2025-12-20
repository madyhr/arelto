// include/raycaster.h
#ifndef RL2_RAYCASTER_H_
#define RL2_RAYCASTER_H_

#include "map.h"
#include "types.h"
namespace rl2 {

struct Raycaster {
  Vector2D ray_start;
  Vector2D ray_dir;
  Vector2D ray_unit_step_size;
  Vector2D ray_length_1d;
};

struct RayHit {
  float distance;
  EntityType entity_type;
};

RayHit
CastRay(const Vector2D &start_pos, const Vector2D &ray_dir,
        const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight> &occupancy_map);
} // namespace rl2

#endif
