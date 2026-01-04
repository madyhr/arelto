// src/observation_manager.cpp
#include "observation_manager.h"
#include <algorithm>
#include <stdexcept>
#include "constants.h"
#include "scene.h"

namespace rl2 {

int ObservationManager::GetObservationSize(const Scene& scene) {
  return kNumRays * 2;
  ;
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

  int idx = 0;

  for (int ray_idx = 0; ray_idx < kNumRays; ++ray_idx) {
    for (int enemy_idx = 0; enemy_idx < kNumEnemies; ++enemy_idx) {
      float val = scene.enemy.ray_caster.ray_hit_distances[ray_idx][enemy_idx] /
                  kMaxRayDistance;

      buffer_ptr[idx++] = std::clamp(val, -1.0f, 1.0f);
    }
  }

  for (int ray_idx = 0; ray_idx < kNumRays; ++ray_idx) {
    for (int enemy_idx = 0; enemy_idx < kNumEnemies; ++enemy_idx) {
      buffer_ptr[idx++] =
          1.0f + static_cast<float>(
                     scene.enemy.ray_caster.ray_hit_types[ray_idx][enemy_idx]);
    }
  }

  return;
};

}  // namespace rl2
