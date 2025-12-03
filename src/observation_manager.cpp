// src/observation_manager.cpp
#include "observation_manager.h"
#include <stdexcept>
#include "scene.h"

namespace rl2 {

int ObservationManager::GetObservationSize(const Scene& scene) {
  return 2 +                  // player position: x,y
         (kNumEnemies * 2) +  // enemy position: x,y
         (kNumEnemies * 2) +  // enemy velocity: x,y
         (kNumEnemies * 2) +  // enemy size: w,h
         (kNumEnemies) +      // enemy health_points
         (kNumEnemies) +      // enemy inv mass
         (kNumEnemies) +      // enemy movement speed
         (kNumEnemies *
          scene.enemy.occupancy_map[0].kTotalCells);  // enemy occupancy map
};

void ObservationManager::FillObservationBuffer(float* buffer_ptr, int buffer_size, const Scene& scene) {

  if (buffer_size != GetObservationSize(scene)) {
    throw std::runtime_error("Buffer size mismatch");
  };

  int idx = 0;

  buffer_ptr[idx++] = scene.player.position_.x;
  buffer_ptr[idx++] = scene.player.position_.y;

  for (const Vector2D& enemy_pos : scene.enemy.position) {
    buffer_ptr[idx++] = enemy_pos.x;
    buffer_ptr[idx++] = enemy_pos.y;
  }

  for (const Vector2D& enemy_pos : scene.enemy.position) {
    buffer_ptr[idx++] = enemy_pos.x;
    buffer_ptr[idx++] = enemy_pos.y;
  }

  for (const Size2D& enemy_size : scene.enemy.sprite_size) {
    buffer_ptr[idx++] = static_cast<float>(enemy_size.width);
    buffer_ptr[idx++] = static_cast<float>(enemy_size.height);
  }

  for (const int& enemy_health : scene.enemy.health_points) {
    buffer_ptr[idx++] = static_cast<float>(enemy_health);
  }

  for (const float& enemy_inv_mass : scene.enemy.inv_mass) {
    buffer_ptr[idx++] = enemy_inv_mass;
  }

  for (const float& enemy_movement_speed : scene.enemy.movement_speed) {
    buffer_ptr[idx++] = enemy_movement_speed;
  }

  for (int i = 0; i < kNumEnemies; ++i) {
    const EntityType* map_data = scene.enemy.occupancy_map[i].Data();
    size_t total_cells = scene.enemy.occupancy_map[i].kTotalCells;

    for (size_t k = 0; k < total_cells; ++k) {
      buffer_ptr[idx++] = static_cast<float>(map_data[k]);
    }
  }

  return;
};

}  // namespace rl2
