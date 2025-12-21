// src/observation_manager.cpp
#include "observation_manager.h"
#include <stdexcept>
#include "constants.h"
#include "scene.h"
#include "types.h"

namespace rl2 {

int ObservationManager::GetObservationSize(const Scene& scene) {
  return 2  // relative position to player
            // 2 +  // enemy position: x,y
            // 2 +  // enemy velocity: x,y
            // 2 +  // enemy size: w,h
            // 1 +  // enemy health_points
            // 1 +  // enemy inv mass
            // 1    // enemy movement speed
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

  for (const Vector2D& enemy_pos : scene.enemy.position) {
    buffer_ptr[idx++] =
        (scene.player.position_.x - enemy_pos.x) / kPositionObservationScale;
  }

  for (const Vector2D& enemy_pos : scene.enemy.position) {
    buffer_ptr[idx++] =
        (scene.player.position_.y - enemy_pos.y) / kPositionObservationScale;
  }

  return;
};

}  // namespace rl2
