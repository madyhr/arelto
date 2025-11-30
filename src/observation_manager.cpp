// src/observation_manager.cpp
#include <stdexcept>
#include "game.h"

namespace rl2 {

int Game::GetObservationSize() {
  return 2 +                  // player position
         (kNumEnemies * 2) +  // enemy position: x,y
         (kNumEnemies * 2) +  // enemy velocity: x,y
         (kNumEnemies * 2) +  // enemy size: w,h
         (kNumEnemies) +      // enemy health_points
         (kNumEnemies) +      // enemy inv mass
         (kNumEnemies) +      // enemy movement speed
         (kNumEnemies *
          enemy_.occupancy_map[0].kTotalCells);  // enemy occupancy map
};

// void Game::FillObservationBuffer(py::array_t<float> buffer) {
void Game::FillObservationBuffer(float* buffer_ptr, int buffer_size) {

  if (buffer_size != GetObservationSize()) {
    throw std::runtime_error("Buffer size mismatch");
  };

  int idx = 0;

  buffer_ptr[idx++] = player_.position_.x;
  buffer_ptr[idx++] = player_.position_.y;

  for (const Vector2D& enemy_pos : enemy_.position) {
    buffer_ptr[idx++] = enemy_pos.x;
    buffer_ptr[idx++] = enemy_pos.y;
  }

  for (const Vector2D& enemy_pos : enemy_.position) {
    buffer_ptr[idx++] = enemy_pos.x;
    buffer_ptr[idx++] = enemy_pos.y;
  }

  for (const Size& enemy_size : enemy_.size) {
    buffer_ptr[idx++] = static_cast<float>(enemy_size.width);
    buffer_ptr[idx++] = static_cast<float>(enemy_size.height);
  }

  for (const int& enemy_health : enemy_.health_points) {
    buffer_ptr[idx++] = static_cast<float>(enemy_health);
  }

  for (const float& enemy_inv_mass : enemy_.inv_mass) {
    buffer_ptr[idx++] = enemy_inv_mass;
  }

  for (const float& enemy_movement_speed : enemy_.movement_speed) {
    buffer_ptr[idx++] = enemy_movement_speed;
  }

  for (int i = 0; i < kNumEnemies; ++i) {
    const EntityType* map_data = enemy_.occupancy_map[i].Data();
    size_t total_cells = enemy_.occupancy_map[i].kTotalCells;

    for (size_t k = 0; k < total_cells; ++k) {
      buffer_ptr[idx++] = static_cast<float>(map_data[k]);
    }
  }

  return;
};

}  // namespace rl2
