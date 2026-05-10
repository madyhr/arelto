// include/scene.h
#ifndef RL2_SCENE_H_
#define RL2_SCENE_H_

#include <algorithm>
#include "config/entity_config.h"
#include "constants/enemy.h"
#include "entity.h"
#include "random.h"
#include "ray_caster.h"
#include "types.h"
#include "upgrades.h"

namespace arelto {

// forward declaration to avoid circular dependency with items.h
class ItemArchive;
enum ItemId : int;
enum class ItemUpgradeType : int;

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
  void Reset(const EntityConfig& entity_config) {

    // Player
    const PlayerConfig& player_config = entity_config.player;
    player.stats_.size = StatsSize(player_config.aspect_ratio);
    player.stats_.size.SetBaseWidth(static_cast<float>(player_config.width));
    player.stats_.max_health.SetBaseValue(player_config.max_health_points);
    player.stats_.health =
        static_cast<int>(player.stats_.max_health.GetValue());
    player.stats_.inv_mass.SetBaseValue(player_config.inv_mass);
    player.stats_.movement_speed.SetBaseValue(player_config.movement_speed);
    player.stats_.global_damage_modifier.SetBaseValue(
        player_config.global_damage_modifier);
    player.stats_.global_cooldown_modifier.SetBaseValue(
        player_config.global_cooldown_modifier);
    player.stats_.level = 0;
    player.stats_.exp_points = 0;
    player.stats_.exp_points_required.SetBaseValue(
        player_config.initial_exp_requirement);
    player.exp_required_scale_ = player_config.exp_required_scale;
    player.invulnerable_window_s_ = player_config.invulnerable_window_s;
    player.is_invulnerable = false;
    player.invulnerable_timer = 0.0f;
    player.is_alive_ = true;
    player.position_ = Vector2D{player_config.spawn_x, player_config.spawn_y};
    player.prev_position_ = player.position_;
    player.last_horizontal_velocity_ = 0.0f;
    player.UpdateAllSpellStats();

    // Enemies
    const EnemyConfig& enemy_config = entity_config.enemy;
    std::fill(enemy.is_alive.begin(), enemy.is_alive.end(), false);
    std::fill(enemy.is_done.begin(), enemy.is_done.end(), false);
    std::fill(enemy.max_health_points.begin(), enemy.max_health_points.end(),
              enemy_config.max_health_points);
    std::fill(enemy.movement_speed.begin(), enemy.movement_speed.end(),
              enemy_config.movement_speed);
    std::fill(enemy.collider.begin(), enemy.collider.end(),
              CreateCenteredCollider(
                  {enemy_config.width, entity_config.enemy.height}));
    std::fill(enemy.sprite_size.begin(), enemy.sprite_size.end(),
              Size2D{enemy_config.width, entity_config.enemy.height});
    std::fill(enemy.inv_mass.begin(), enemy.inv_mass.end(),
              enemy_config.inv_mass);
    std::fill(enemy.attack_cooldown_s.begin(), enemy.attack_cooldown_s.end(),
              enemy_config.attack_cooldown_s);
    std::fill(enemy.attack_cooldown_timer.begin(),
              enemy.attack_cooldown_timer.end(), 0.0f);
    std::fill(enemy.attack_damage.begin(), enemy.attack_damage.end(),
              enemy_config.attack_damage);
    SpawnAllEnemies(enemy, player);
    SetupEnemyRayCasterPattern(enemy.ray_caster);

    // Add slight variation to each enemy to make it more interesting.
    for (int i = 0; i < kNumEnemies; ++i) {
      enemy.movement_speed[i] += static_cast<float>(GenerateRandomInt(1, 100));
      int random_width = GenerateRandomInt(1, 50);
      int random_height = GenerateRandomInt(1, 50);
      enemy.sprite_size[i].width += random_width;
      enemy.sprite_size[i].height += random_height;
      // TODO: Figure out a more maintainable solution to collider + sprite size randomization.
      enemy.collider[i].offset.x += 0.5f * static_cast<float>(random_width);
      enemy.collider[i].offset.y += 0.5f * static_cast<float>(random_height);
      enemy.collider[i].size.width += random_width;
      enemy.collider[i].size.height += random_height;
    };

    projectiles.ResetAllProjectiles();
    exp_gem.ResetAllExpGems();
    chest.ResetAllChests();
    player.inventory_.clear();
  };
};

}  // namespace arelto

#endif
