// include/scene.h
#ifndef RL2_SCENE_H_
#define RL2_SCENE_H_

#include "constants/enemy.h"
#include "constants/player.h"
#include "entity.h"
#include "random.h"
#include "ray_caster.h"
#include "types.h"

namespace rl2 {

struct Scene {

  Player player;
  Enemy enemy;
  Projectiles projectiles;
  ExpGem exp_gem;
  FixedMap<kOccupancyMapWidth, kOccupancyMapHeight> occupancy_map;

  void Reset() {

    // -- Player
    player.collider_ =
        Collider{{kPlayerColliderOffsetX, kPlayerColliderOffsetY},
                 {kPlayerColliderWidth, kPlayerColliderHeight}};
    player.stats_.sprite_size = Size2D{kPlayerSpriteWidth, kPlayerSpriteHeight};
    player.stats_.max_health = kPlayerInitMaxHealth;
    player.stats_.health = player.stats_.max_health;
    player.stats_.inv_mass = kPlayerInvMass;
    player.stats_.movement_speed = kPlayerSpeed;
    player.stats_.level = 0;
    player.stats_.exp_points = 0;
    player.stats_.exp_points_required = kPlayerInitialExpRequirement;
    player.position_ = Vector2D{kPlayerInitX, kPlayerInitY};
    player.prev_position_ = player.position_;
    player.UpdateAllSpellStats();

    // -- Enemies
    std::fill(enemy.is_alive.begin(), enemy.is_alive.end(), false);
    std::fill(enemy.is_done.begin(), enemy.is_done.end(), false);
    std::fill(enemy.movement_speed.begin(), enemy.movement_speed.end(),
              kEnemySpeed);
    std::fill(enemy.collider.begin(), enemy.collider.end(),
              Collider{{kEnemyColliderOffsetX, kEnemyColliderOffsetY},
                       {kEnemyColliderWidth, kEnemyColliderHeight}});
    std::fill(enemy.sprite_size.begin(), enemy.sprite_size.end(),
              Size2D{kEnemySpriteWidth, kEnemySpriteHeight});
    std::fill(enemy.inv_mass.begin(), enemy.inv_mass.end(), kEnemyInvMass);
    std::fill(enemy.attack_cooldown.begin(), enemy.attack_cooldown.end(), 0.0f);
    std::fill(enemy.attack_damage.begin(), enemy.attack_damage.end(), 1);
    RespawnEnemy(enemy, player);
    SetupEnemyRayCasterPattern(enemy.ray_caster);

    // Add slight variation to each enemy to make it more interesting.
    for (int i = 0; i < kNumEnemies; ++i) {
      enemy.movement_speed[i] += GenerateRandomInt(1, 100);
      int random_width = GenerateRandomInt(1, 50);
      int random_height = GenerateRandomInt(1, 50);
      enemy.sprite_size[i].width += random_width;
      enemy.sprite_size[i].height += random_height;
      // TODO: Figure out a more maintainable solution to collider + sprite size randomization.
      enemy.collider[i].offset.x += 0.5f * random_width;
      enemy.collider[i].offset.y += 0.5f * random_height;
      enemy.collider[i].size.width += random_width;
      enemy.collider[i].size.height += random_height;
    };

    // -- Projectiles
    projectiles.ResetAllProjectiles();

    // -- Exp Gems
    exp_gem.ResetAllExpGems();
  };

  void SpawnExpGem() {
    for (int i = 0; i < kNumEnemies; ++i) {
      if (enemy.is_done[i]) {
        ExpGemData gem_data = {ExpGemType::small, enemy.position[i],
                               enemy.prev_position[i], enemy.collider[i],
                               enemy.sprite_size[i]};
        exp_gem.AddExpGem(gem_data);
      }
    }
  };
};

}  // namespace rl2

#endif
