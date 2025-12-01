// include/scene.h
#ifndef RL2_SCENE_H_
#define RL2_SCENE_H_

#include "entity.h"
#include "random.h"

namespace rl2 {

struct Scene {

  Player player;
  Enemy enemy;
  Projectiles projectiles;
  FixedMap<kOccupancyMapWidth, kOccupancyMapHeight> occupancy_map;

  void Reset() {

    // Player
    player.stats_.size = {kPlayerWidth, kPlayerHeight};
    player.stats_.inv_mass = kPlayerInvMass;
    player.position_ = {kPlayerInitX, kPlayerInitY};
    player.stats_.movement_speed = kPlayerSpeed;
    player.UpdateAllSpellStats();

    // Enemies
    std::fill(enemy.is_alive.begin(), enemy.is_alive.end(), false);
    std::fill(enemy.movement_speed.begin(), enemy.movement_speed.end(),
              kEnemySpeed);
    std::fill(enemy.size.begin(), enemy.size.end(),
              Size{kEnemyHeight, kEnemyWidth});
    std::fill(enemy.inv_mass.begin(), enemy.inv_mass.end(), kEnemyInvMass);
    RespawnEnemy(enemy, player);

    // Add slight variation to each enemy to make it more interesting.
    for (int i = 0; i < kNumEnemies; ++i) {
      enemy.movement_speed[i] += GenerateRandomInt(1, 100);
      enemy.size[i].height += GenerateRandomInt(1, 50);
      enemy.size[i].width += GenerateRandomInt(1, 50);
    };
  };
};

}  // namespace rl2

#endif
