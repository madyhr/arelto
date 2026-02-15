// src/ray_caster.cpp
#include "ray_caster.h"
#include <cstdint>
#include "constants/map.h"
#include "types.h"

namespace arelto {

// This function assumes that the occupancy map is surrounded by grid cells
// that have an EntityType other than None.
DualRayHit CastRay(
    const Vector2D& start_pos, const Vector2D& ray_dir,
    const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map) {

  int step_x, step_y;
  float side_dist_x, side_dist_y;
  bool hit_side_x;

  RayHit blocking_hit = {0.0f, EntityType::None};
  RayHit non_blocking_hit = {0.0f, EntityType::None};

  Vector2D grid_pos = WorldToGrid(start_pos);
  int map_grid_x = static_cast<int>(grid_pos.x);
  int map_grid_y = static_cast<int>(grid_pos.y);

  float delta_dist_x = std::abs(1 / ray_dir.x);
  float delta_dist_y = std::abs(1 / ray_dir.y);

  if (ray_dir.x < 0) {
    step_x = -1;
    side_dist_x = (grid_pos.x - map_grid_x) * delta_dist_x;
  } else {
    step_x = 1;
    side_dist_x = (map_grid_x + 1 - grid_pos.x) * delta_dist_x;
  }

  if (ray_dir.y < 0) {
    step_y = -1;
    side_dist_y = (grid_pos.y - map_grid_y) * delta_dist_y;
  } else {
    step_y = 1;
    side_dist_y = (map_grid_y + 1 - grid_pos.y) * delta_dist_y;
  }

  bool blocking_found = false;
  bool non_blocking_found = false;

  while (!blocking_found) {

    if (side_dist_x < side_dist_y) {
      map_grid_x += step_x;
      side_dist_x += delta_dist_x;
      hit_side_x = true;
    } else {
      map_grid_y += step_y;
      side_dist_y += delta_dist_y;
      hit_side_x = false;
    }

    // Use GetMask for efficient multi-type checking
    uint16_t mask = occupancy_map.GetMask(map_grid_x, map_grid_y);

    // Check for non-blocking (record closest only)
    // TODO: Extend ray caster to allow for any number of non-blocking hits?
    if (!non_blocking_found && (mask & kMaskTypeProjectile)) {
      float dist;
      if (hit_side_x) {
        dist = side_dist_x - delta_dist_x;
      } else {
        dist = side_dist_y - delta_dist_y;
      }
      non_blocking_hit = {GridToWorld(dist), EntityType::projectile};
      non_blocking_found = true;
    }

    // Check for blockers
    if (mask & kMaskRayHitBlockingTypes) {
      EntityType type = MaskToEntityTypePrioritized(mask);
      float dist;
      if (hit_side_x) {
        dist = side_dist_x - delta_dist_x;
      } else {
        dist = side_dist_y - delta_dist_y;
      }
      blocking_hit = {GridToWorld(dist), type};
      blocking_found = true;
    }
  }

  return {blocking_hit, non_blocking_hit};
};

void SetupEnemyRayCasterPattern(EnemyRayCaster& ray_caster) {

  size_t num_rays = ray_caster.pattern.ray_dir.size();

  for (int i = 0; i < num_rays; ++i) {
    float degree = i * (360.0f / num_rays);
    ray_caster.pattern.ray_dir[i] = {std::cos(Deg2Rad(degree)),
                                     std::sin(Deg2Rad(degree))};
  }
};

bool IsEntityTypePresent(const RayHistoryTypes& ray_hit_types,
                         size_t history_idx, size_t enemy_idx,
                         EntityType target) {
  const auto& history_frame = ray_hit_types[history_idx];

  for (size_t r = 0; r < kNumRays; ++r) {
    if (history_frame[r][enemy_idx] == target) {
      return true;
    }
  }
  return false;
}

}  // namespace arelto
