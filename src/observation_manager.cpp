// src/observation_manager.cpp
#include "observation_manager.h"
#include <algorithm>
#include <stdexcept>
#include "constants/enemy.h"
#include "constants/ray_caster.h"
#include "scene.h"

namespace arelto {

int ObservationManager::GetObservationSize() {
  return 2 * kNumRays * kNumRayTypes * kRayHistoryLength;
};

// Function that fills an observation buffer to be used for learning.
// Each observation is looped over separately to ensure that
// transposing the array will transform the unflattened size from
// (num_features, num_enemies) to (num_enemies, num_features).
void ObservationManager::FillObservationBuffer(float* buffer_ptr,
                                               int buffer_size,
                                               const Scene& scene) {

  if (buffer_size != kNumEnemies * GetObservationSize()) {
    throw std::runtime_error("Buffer size mismatch");
  };

  int obs_buffer_idx = 0;

  // Current head points to the next write slot, which is also the oldest slot
  // We want to iterate from oldest to newest.
  int current_head = scene.enemy.ray_caster.history_idx;

  // Fill all blocking distances for all frames (oldest -> newest)
  for (int i = 0; i < kRayHistoryLength; ++i) {
    int history_buffer_idx = (current_head + i) % kRayHistoryLength;

    for (int ray_idx = 0; ray_idx < kNumRays; ++ray_idx) {
      for (int enemy_idx = 0; enemy_idx < kNumEnemies; ++enemy_idx) {

        buffer_ptr[obs_buffer_idx++] =
            scene.enemy.ray_caster
                .ray_hit_distances[history_buffer_idx][ray_idx][enemy_idx];
      }
    }
  }

  // Fill all non-blocking distances for all frames (oldest -> newest)
  for (int i = 0; i < kRayHistoryLength; ++i) {
    int history_buffer_idx = (current_head + i) % kRayHistoryLength;

    for (int ray_idx = 0; ray_idx < kNumRays; ++ray_idx) {
      for (int enemy_idx = 0; enemy_idx < kNumEnemies; ++enemy_idx) {

        buffer_ptr[obs_buffer_idx++] =
            scene.enemy.ray_caster
                .non_blocking_ray_hit_distances[history_buffer_idx][ray_idx]
                                               [enemy_idx];
      }
    }
  }

  // Fill all blocking types for all frames (oldest -> newest)
  for (int i = 0; i < kRayHistoryLength; ++i) {
    int history_buffer_idx = (current_head + i) % kRayHistoryLength;

    for (int ray_idx = 0; ray_idx < kNumRays; ++ray_idx) {
      for (int enemy_idx = 0; enemy_idx < kNumEnemies; ++enemy_idx) {
        buffer_ptr[obs_buffer_idx++] = static_cast<float>(
            scene.enemy.ray_caster
                .ray_hit_types[history_buffer_idx][ray_idx][enemy_idx]);
      }
    }
  }

  // Fill all non-blocking types for all frames (oldest -> newest)
  for (int i = 0; i < kRayHistoryLength; ++i) {
    int history_buffer_idx = (current_head + i) % kRayHistoryLength;

    for (int ray_idx = 0; ray_idx < kNumRays; ++ray_idx) {
      for (int enemy_idx = 0; enemy_idx < kNumEnemies; ++enemy_idx) {
        buffer_ptr[obs_buffer_idx++] = static_cast<float>(
            scene.enemy.ray_caster
                .non_blocking_ray_hit_types[history_buffer_idx][ray_idx]
                                           [enemy_idx]);
      }
    }
  }

  return;
};

void ObservationManager::UpdateObservations(Scene& scene) {
  if (physics_tick_count_ % kOccupancyMapTimeDecimation == 0) {
    UpdateWorldOccupancyMap(scene.occupancy_map, scene.player, scene.enemy,
                            scene.projectiles);
    UpdateEnemyRayCaster(scene.enemy, scene.occupancy_map);
  }
  physics_tick_count_++;
}

void ObservationManager::UpdateWorldOccupancyMap(
    FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map,
    Player& player, Enemy& enemy, Projectiles& projectiles) {
  occupancy_map.Clear();

  // Helper lambda to use an entity's AABB to set the entity type on the
  // world occupancy map accordingly.
  auto MarkOccupancy = [&](Vector2D pos, Collider col, EntityType type) {
    Vector2D center = pos + col.offset;
    AABB aabb = GetCollisionAABB(center, col.size);

    Vector2D grid_min = WorldToGrid(Vector2D{aabb.min_x, aabb.min_y});
    Vector2D grid_max = WorldToGrid(Vector2D{aabb.max_x, aabb.max_y});

    int start_x = static_cast<int>(grid_min.x);
    int start_y = static_cast<int>(grid_min.y);
    int end_x = static_cast<int>(grid_max.x);
    int end_y = static_cast<int>(grid_max.y);

    start_x = std::max(0, start_x);
    start_y = std::max(0, start_y);
    end_x = std::min(kOccupancyMapWidth - 1, end_x);
    end_y = std::min(kOccupancyMapHeight - 1, end_y);

    for (int x = start_x; x <= end_x; ++x) {
      for (int y = start_y; y <= end_y; ++y) {
        occupancy_map.Add(x, y, type);
      }
    }
  };

  MarkOccupancy(player.position_, player.stats_.size.GetCollider(),
                player.entity_type_);

  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemy.is_alive[i]) {
      MarkOccupancy(enemy.position[i], enemy.collider[i], enemy.entity_type);
    }
  }

  const size_t num_proj = projectiles.GetNumProjectiles();
  for (int i = 0; i < num_proj; ++i) {
    MarkOccupancy(projectiles.position_[i], projectiles.collider_[i],
                  projectiles.entity_type_);
  };

  // A border is added around the map to ensure that a ray always hits a grid
  // cell with a EntityType other than None and thereby always terminates.
  occupancy_map.AddBorder(EntityType::terrain);
}

void ObservationManager::UpdateEnemyRayCaster(
    Enemy& enemy,
    const FixedMap<kOccupancyMapWidth, kOccupancyMapHeight>& occupancy_map) {
  int history_idx = enemy.ray_caster.history_idx;

#pragma omp parallel for
  for (int ray_idx = 0; ray_idx < kNumRays; ++ray_idx) {
    Vector2D dir = enemy.ray_caster.pattern.ray_dir[ray_idx];

    for (int enemy_idx = 0; enemy_idx < kNumEnemies; ++enemy_idx) {
      if (!enemy.is_alive[enemy_idx]) {
        continue;
      }

      Vector2D center =
          enemy.position[enemy_idx] + enemy.collider[enemy_idx].offset;

      float half_w =
          static_cast<float>(enemy.collider[enemy_idx].size.width) * 0.5f;
      float half_h =
          static_cast<float>(enemy.collider[enemy_idx].size.height) * 0.5f;

      // a small offset is added to ensure the rays do not clip the corners
      // of the collider. Note: this does add blind spots.
      float ray_offset = std::max(half_h, half_w) + kMinRayDistance;
      Vector2D start_pos = center + dir * ray_offset;

      Vector2D grid_pos = WorldToGrid(start_pos);

      // Before actually casting a ray, since we cast the ray offset from the
      // center position, we need to check if we are about to cast through
      // terrain which could lead to a ray going OOB. In that case, we should
      // just skip the ray casting altogether and we can set the distance to 0.
      bool out_of_bounds = grid_pos.x >= kOccupancyMapWidth - 1 ||
                           grid_pos.y >= kOccupancyMapHeight - 1 ||
                           grid_pos.x <= 1 || grid_pos.y <= 1;

      DualRayHit dual_hit;
      if (out_of_bounds) {
        dual_hit.blocking_hit = {0.0f, EntityType::terrain};
        dual_hit.non_blocking_hit = {0.0f, EntityType::None};
      } else {
        dual_hit = CastRay({start_pos, dir}, occupancy_map);
      }

      enemy.ray_caster.ray_hit_distances[history_idx][ray_idx][enemy_idx] =
          dual_hit.blocking_hit.distance;
      enemy.ray_caster.ray_hit_types[history_idx][ray_idx][enemy_idx] =
          dual_hit.blocking_hit.entity_type;

      enemy.ray_caster
          .non_blocking_ray_hit_distances[history_idx][ray_idx][enemy_idx] =
          dual_hit.non_blocking_hit.distance;
      enemy.ray_caster
          .non_blocking_ray_hit_types[history_idx][ray_idx][enemy_idx] =
          dual_hit.non_blocking_hit.entity_type;
    }
  }

  enemy.ray_caster.history_idx =
      (enemy.ray_caster.history_idx + 1) % kRayHistoryLength;
}

}  // namespace arelto
