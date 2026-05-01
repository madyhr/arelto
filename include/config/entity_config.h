#ifndef RL2_CONFIG_ENTITY_CONFIG_H_
#define RL2_CONFIG_ENTITY_CONFIG_H_

#include <cstdint>
namespace arelto {

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
  EnemyConfig enemy;
};

EntityConfig MakeDefaultEntityConfig();

}  // namespace arelto
#endif
