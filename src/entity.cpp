// src/entity.cpp
#include "entity.h"
#include "types.h"

namespace rl2 {

void Entity::update_aabb() {
  aabb = {position.x, position.y, position.x + stats.size.width,
          position.y + stats.size.height, 0};
};

void Projectiles::add_projectile(ProjectileData proj){
  owner_id.push_back(proj.owner_id);
  position.push_back(proj.position);
  velocity.push_back(proj.velocity);
  speed.push_back(proj.speed);
  size.push_back(proj.size);
  inv_mass.push_back(proj.inv_mass);
};

void Projectiles::destroy_projectile(int idx){
  size_t last_idx = position.size() - 1;
  if (idx != last_idx){
    owner_id[idx] = std::move(owner_id[last_idx]);
    position[idx] = std::move(position[last_idx]);
    velocity[idx] = std::move(velocity[last_idx]);
    speed[idx] = std::move(speed[last_idx]);
    size[idx] = std::move(size[last_idx]);
    inv_mass[idx] = std::move(inv_mass[last_idx]);
  };
  owner_id.pop_back();
  position.pop_back();
  velocity.pop_back();
  speed.pop_back();
  size.pop_back();
  inv_mass.pop_back();

};

}  // namespace rl2
