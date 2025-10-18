// src/entity.cpp
#include "entity.h"
#include "constants.h"

namespace rl2 {

void Entity::update_aabb() {
  aabb = {position.x, position.y, position.x + stats.size.width,
          position.y + stats.size.height, 0};
};
}  // namespace rl2
