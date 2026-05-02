#ifndef RL2_CONFIG_ENTITY_CONFIG_H_
#define RL2_CONFIG_ENTITY_CONFIG_H_

#include <cstdint>
namespace arelto {

struct PlayerConfig {
  float max_health_points = 100.0f;
  float spawn_x = 5000.0f;
  float spawn_y = 5000.0f;
  float movement_speed = 200.0f;
  uint32_t width = 60;
  uint32_t height = 75;
  float inv_mass = 0.01f;
  float initial_exp_requirement = 10.0f;
  float exp_required_scale = 1.1f;
  float invulnerable_window_s = 0.1f;
};

struct EnemyConfig {
  int max_health_points = 10;
  int attack_damage = 1;
  float attack_cooldown_s = 0.1f;
  float movement_speed = 40.0f;
  uint32_t width = 42;
  uint32_t height = 50;
  float inv_mass = 0.1f;
};

struct EntityConfig {
  PlayerConfig player;
  EnemyConfig enemy;
};

EntityConfig MakeDefaultEntityConfig();

}  // namespace arelto
#endif
