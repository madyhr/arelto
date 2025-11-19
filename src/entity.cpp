// src/entity.cpp
#include "entity.h"
#include "types.h"

namespace rl2 {

void Entity::UpdateAABB() {
  aabb_ = {position_.x, position_.y, position_.x + stats_.size.width,
           position_.y + stats_.size.height, 0};
};

void Projectiles::AddProjectile(ProjectileData proj) {
  owner_ids_.push_back(proj.owner_id);
  positions_.push_back(proj.position);
  velocities_.push_back(proj.velocity);
  speeds_.push_back(proj.speed);
  sizes_.push_back(proj.size);
  inv_masses_.push_back(proj.inv_mass);
};

void Projectiles::DestroyProjectile(int idx) {
  size_t last_idx = positions_.size() - 1;
  if (idx != last_idx) {
    owner_ids_[idx] = std::move(owner_ids_[last_idx]);
    positions_[idx] = std::move(positions_[last_idx]);
    velocities_[idx] = std::move(velocities_[last_idx]);
    speeds_[idx] = std::move(speeds_[last_idx]);
    sizes_[idx] = std::move(sizes_[last_idx]);
    inv_masses_[idx] = std::move(inv_masses_[last_idx]);
  };
  owner_ids_.pop_back();
  positions_.pop_back();
  velocities_.pop_back();
  speeds_.pop_back();
  sizes_.pop_back();
  inv_masses_.pop_back();
};

}  // namespace rl2
