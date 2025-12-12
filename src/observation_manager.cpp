// src/observation_manager.cpp
#include "observation_manager.h"
#include <stdexcept>
#include "constants.h"
#include "scene.h"
#include "types.h"

namespace rl2 {

int ObservationManager::GetObservationSize(const Scene& scene) {
  return 1 +  // distance to player
         2 +  // enemy position: x,y
         2 +  // enemy velocity: x,y
         2 +  // enemy size: w,h
         1 +  // enemy health_points
         1 +  // enemy inv mass
         1    // enemy movement speed
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
        (scene.player.position_ - enemy_pos).Norm() * kInvMapMaxDistance;
  }

  for (const Vector2D& enemy_pos : scene.enemy.position) {
    buffer_ptr[idx++] = enemy_pos.x / kMapWidth;
  }

  for (const Vector2D& enemy_pos : scene.enemy.position) {
    buffer_ptr[idx++] = enemy_pos.y / kMapHeight;
  }

  for (const Vector2D& enemy_vel : scene.enemy.velocity) {
    buffer_ptr[idx++] = enemy_vel.x;
  }

  for (const Vector2D& enemy_vel : scene.enemy.velocity) {
    buffer_ptr[idx++] = enemy_vel.y;
  }

  for (const Collider& enemy_collider : scene.enemy.collider) {
    buffer_ptr[idx++] = static_cast<float>(enemy_collider.size.width) / kEnemyColliderWidth;
  }

  for (const Collider& enemy_collider : scene.enemy.collider) {
    buffer_ptr[idx++] = static_cast<float>(enemy_collider.size.height) / kEnemyColliderHeight;
  }

  for (const int& enemy_health : scene.enemy.health_points) {
    buffer_ptr[idx++] = static_cast<float>(enemy_health) / kEnemyHealth;
  }

  for (const float& enemy_inv_mass : scene.enemy.inv_mass) {
    buffer_ptr[idx++] = enemy_inv_mass / kEnemyInvMass;
  }

  for (const float& enemy_movement_speed : scene.enemy.movement_speed) {
    buffer_ptr[idx++] = enemy_movement_speed / kEnemySpeed;
  }

  // for (int i = 0; i < kNumEnemies; ++i) {
  //   const EntityType* map_data = scene.enemy.occupancy_map[i].Data();
  //   size_t total_cells = scene.enemy.occupancy_map[i].kTotalCells;
  //
  //   for (size_t k = 0; k < total_cells; ++k) {
  //     buffer_ptr[idx++] = static_cast<float>(map_data[k]);
  //   }
  // }

  return;
};

}  // namespace rl2
