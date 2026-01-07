// src/action_manager.cpp
#include "action_manager.h"
#include <stdexcept>
#include "constants/enemy.h"
#include "scene.h"

namespace rl2 {

int ActionManager::GetActionSize(const Scene& scene) {
  return 2;  // enemy velocity (x,y)
};

// Function that reads a buffer with int corresponding to enemy velocities
// and sets them accordingly: 0 = -1.0, 1 = 0.0f, 2 = 1.0f.
// Each action is looped over separately as the unflattened size of the action
// array is (num_actions, num_enemies).
void ActionManager::ReadActionBuffer(int* buffer_ptr, int buffer_size,
                                     Scene& scene) {
  if (buffer_size != (kNumEnemies * GetActionSize(scene))) {
    throw std::runtime_error("Action buffer size mismatch");
  };

  int idx = 0;

  for (Vector2D& enemy_vel : scene.enemy.velocity) {
    enemy_vel.x = static_cast<float>(buffer_ptr[idx++] - 1);
  }

  for (Vector2D& enemy_vel : scene.enemy.velocity) {
    enemy_vel.y = static_cast<float>(buffer_ptr[idx++] - 1);
  }

  return;
};

}  // namespace rl2
