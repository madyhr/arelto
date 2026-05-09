#ifndef RL2_CONFIG_ENTITY_CONFIG_H_
#define RL2_CONFIG_ENTITY_CONFIG_H_

#include <array>
#include <cstdint>
#include "types.h"
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
  float global_damage_modifier = 1.0f;
  float global_cooldown_modifier = 1.0f;
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

struct ExpGemRarityConfig {
  int exp_value = 1;
  uint32_t width = 25;
  uint32_t height = 33;
};

inline std::array<ExpGemRarityConfig, Rarity::Count>
MakeDefaultExpGemRarities() {
  std::array<ExpGemRarityConfig, Rarity::Count> rarities{};
  rarities[Rarity::common] = {1, 25, 33};
  rarities[Rarity::rare] = {2, 30, 40};
  rarities[Rarity::epic] = {4, 35, 45};
  rarities[Rarity::legendary] = {8, 45, 60};
  return rarities;
}

struct ExpGemConfig {
  float inv_mass = 1.0f;
  std::array<ExpGemRarityConfig, Rarity::Count> rarities =
      MakeDefaultExpGemRarities();
};

struct ChestConfig {
  float spawn_chance = 0.99f;
  float gem_min_separation = 50.0f;
  uint32_t width = 70;
  uint32_t height = 55;
  float inv_mass = 1.0f;
};

struct EntityConfig {
  PlayerConfig player;
  EnemyConfig enemy;
  ExpGemConfig exp_gem;
  ChestConfig chest;
};

EntityConfig MakeDefaultEntityConfig();

}  // namespace arelto
#endif
