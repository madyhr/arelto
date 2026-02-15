// src/observation_manager.cpp
#include "observation_manager.h"
#include <stdexcept>
#include "constants/enemy.h"
#include "constants/ray_caster.h"
#include "scene.h"

namespace arelto {

int ObservationManager::GetObservationSize(const Scene& scene) {
  return 2 * kNumRays * kNumRayTypes * kRayHistoryLength;
};

// Function that fills an observation buffer to be used for learning.
// Each observation is looped over separately to ensure that
// transposing the array will transform the unflattened size from
// (num_features, num_enemies) to (num_enemies, num_features).
void ObservationManager::FillObservationBuffer(float* buffer_ptr,
                                               int buffer_size,
                                               const Scene& scene) {

  if (buffer_size != kNumEnemies * GetObservationSize(scene)) {
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

}  // namespace arelto
