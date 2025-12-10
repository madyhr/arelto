// src/action_manager.cpp
#include "action_manager.h"
#include <stdexcept>
#include "constants.h"
#include "scene.h"

namespace rl2 {

int ActionManager::GetActionSize(const Scene& scene) {
  return 2;  // enemy velocity (x,y)
};

// Function that reads a buffer with floats corresponding to enemy velocities
// and sets them accordingly.
// Each action is looped over separately as the unflattened size of the action
// array is (num_actions, num_enemies). 
void ActionManager::ReadActionBuffer(float* buffer_ptr, int buffer_size,
                                     Scene& scene) {
  if (buffer_size != (kNumEnemies * GetActionSize(scene))) {
    throw std::runtime_error("Action buffer size mismatch");
  };

  int idx = 0;

  for (Vector2D& enemy_vel : scene.enemy.velocity) {
    enemy_vel.x = buffer_ptr[idx++];
  }

  for (Vector2D& enemy_vel : scene.enemy.velocity) {
    enemy_vel.y = buffer_ptr[idx++];
  }

  return;
};

}  // namespace rl2
