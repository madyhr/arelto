// include/scene.h
#ifndef RL2_SCENE_H_
#define RL2_SCENE_H_

#include "constants/chest.h"
#include "constants/enemy.h"
#include "constants/exp_gem.h"
#include "constants/player.h"
#include "entity.h"
#include "items.h"
#include "random.h"
#include "ray_caster.h"
#include "types.h"
#include "upgrades.h"

namespace arelto {

struct Scene {

  Player player;
  Enemy enemy;
  Projectiles projectiles;
  ExpGem exp_gem;
  Chest chest;
  FixedMap<kOccupancyMapWidth, kOccupancyMapHeight> occupancy_map;
  UpgradeOptions level_up_options;
  UpgradeOptions item_options;
  ItemArchive* item_archive;

  void Reset() {

    // Player
    player.collider_ =
        Collider{{kPlayerColliderOffsetX, kPlayerColliderOffsetY},
                 {kPlayerColliderWidth, kPlayerColliderHeight}};
    player.stats_.sprite_size = Size2D{kPlayerSpriteWidth, kPlayerSpriteHeight};
    player.stats_.max_health.SetBaseValue(kPlayerInitMaxHealth);
    player.stats_.health =
        static_cast<int>(player.stats_.max_health.GetValue());
    player.stats_.inv_mass.SetBaseValue(kPlayerInvMass);
    player.stats_.movement_speed.SetBaseValue(kPlayerSpeed);
    player.stats_.level = 0;
    player.stats_.exp_points = 0;
    player.stats_.exp_points_required.SetBaseValue(
        kPlayerInitialExpRequirement);
    player.position_ = Vector2D{kPlayerInitX, kPlayerInitY};
    player.prev_position_ = player.position_;
    player.last_horizontal_velocity_ = 0.0f;
    player.UpdateAllSpellStats();

    // Enemies
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

    // Projectiles
    projectiles.ResetAllProjectiles();

    // Exp Gems
    exp_gem.ResetAllExpGems();

    // Chests
    chest.ResetAllChests();
  };

  void SpawnExpGem() {
    for (int i = 0; i < kNumEnemies; ++i) {
      if (enemy.is_done[i]) {
        Vector2D enemy_centroid =
            GetCentroid(enemy.position[i], enemy.collider[i].size);

        Rarity random_rarity = static_cast<Rarity>(GenerateRandomInt(0, 3));
        ExpGemData gem_data = {random_rarity, enemy_centroid, enemy_centroid,
                               kExpGemCollider[random_rarity],
                               kExpGemSpriteSize[random_rarity]};
        exp_gem.AddExpGem(gem_data);
      }
    }
  };

  void SpawnChest() {
    for (int i = 0; i < kNumEnemies; ++i) {
      if (enemy.is_done[i]) {
        float roll = static_cast<float>(GenerateRandomInt(0, 99)) / 100.0f;
        if (roll < kChestSpawnChance) {
          Vector2D enemy_centroid =
              GetCentroid(enemy.position[i], enemy.collider[i].size);
          ChestData chest_data = {enemy_centroid, enemy_centroid,
                                  kChestCollider, kChestSpriteSize};
          chest.AddChest(chest_data);
        }
      }
    }
  };
};

}  // namespace arelto

#endif
