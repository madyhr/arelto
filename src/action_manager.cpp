// src/action_manager.cpp
#include "action_manager.h"
#include <stdexcept>
#include "constants.h"
#include "scene.h"

namespace rl2 {

int ActionManager::GetActionSize(const Scene& scene) {
  return 2;  // enemy velocity (x,y)
};

void ActionManager::ReadActionBuffer(float* buffer_ptr, int buffer_size,
                                     Scene& scene) {
  if (buffer_size != (kNumEnemies * GetActionSize(scene))) {
    throw std::runtime_error("Action buffer size mismatch");
  };

  int idx = 0;

  for (Vector2D& enemy_vel : scene.enemy.velocity) {
    enemy_vel.x = buffer_ptr[idx++];
    enemy_vel.y = buffer_ptr[idx++];
  }

  return;
};

}  // namespace rl2
