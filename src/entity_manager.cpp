// src/entity_manager.cpp
#include "entity_manager.h"
#include "config/entity_config_yaml.h"  // IWYU pragma: keep
#include "event_manager.h"
#include "random.h"
#include "types.h"

namespace arelto {

EntityManager::EntityManager() {}
EntityManager::~EntityManager() {}

void EntityManager::LoadEntityConfig() {
  entity_config_ = MakeDefaultEntityConfig();
  config_manager_.LoadConfigSectionOrDefault("entity.player",
                                             "assets/config/entity/player.yaml",
                                             entity_config_.player);
  config_manager_.LoadConfigSectionOrDefault(
      "entity.enemy", "assets/config/entity/enemy.yaml", entity_config_.enemy);
  config_manager_.LoadConfigSectionOrDefault(
      "entity.exp_gem", "assets/config/entity/exp_gem.yaml",
      entity_config_.exp_gem);
  config_manager_.LoadConfigSectionOrDefault(
      "entity.chest", "assets/config/entity/chest.yaml", entity_config_.chest);
}

void EntityManager::Initialize(EventManager& event_manager) {
  LoadEntityConfig();
  event_manager_ = &event_manager;

  event_manager.Subscribe<EnemyKilledEvent>(
      [this](const EnemyKilledEvent& event, EventContext& event_context) {
        OnEnemyKilled(event, event_context);
      });

  event_manager.Subscribe<PlayerExpGemCollisionEvent>(
      [this](const PlayerExpGemCollisionEvent& event,
             EventContext& event_context) {
        OnPlayerExpGemCollision(event, event_context);
      });

  event_manager.Subscribe<PlayerChestCollisionEvent>(
      [this](const PlayerChestCollisionEvent& event,
             EventContext& event_context) {
        OnPlayerChestCollision(event, event_context);
      });

  event_manager.Subscribe<PlayerDamagedEvent>(
      [this](const PlayerDamagedEvent& event, EventContext& event_context) {
        OnPlayerDamaged(event, event_context);
      });

  event_manager.Subscribe<PlayerEnemyCollisionEvent>(
      [this](const PlayerEnemyCollisionEvent& event,
             EventContext& event_context) {
        OnPlayerEnemyCollision(event, event_context);
      });

  event_manager.Subscribe<EnemyDamagedEvent>(
      [this](const EnemyDamagedEvent& event, EventContext& event_context) {
        OnEnemyDamaged(event, event_context);
      });

  event_manager.Subscribe<EnemyProjectileCollisionEvent>(
      [this](const EnemyProjectileCollisionEvent& event,
             EventContext& event_context) {
        OnEnemyProjectileCollision(event, event_context);
      });
}

// ---------------------------------------------------------------------------
// Event Handlers
// ---------------------------------------------------------------------------
namespace {

Rarity SampleRandomExpGemRarity(const EntityConfig& entity_config) {
  std::vector<float> weights;
  weights.reserve(Rarity::Count);
  for (int i = 0; i < Rarity::Count; ++i) {
    weights.push_back(entity_config.exp_gem.rarities[i].spawn_weighting);
  }

  return static_cast<Rarity>(SampleFromDiscreteDist(weights));
};
}  //namespace

void EntityManager::OnEnemyKilled(const EnemyKilledEvent& event,
                                  EventContext& context) {
  Vector2D centroid =
      GetCentroid(context.scene.enemy.position[event.enemy_idx],
                  context.scene.enemy.collider[event.enemy_idx].size);

  float chest_roll = static_cast<float>(GenerateRandomInt(0, 99)) / 100.0f;
  bool chest_will_spawn = chest_roll < entity_config_.chest.spawn_chance;

  float spawn_offset =
      chest_will_spawn ? entity_config_.chest.gem_min_separation * 0.5f : 0.0f;
  Vector2D gem_position = {centroid.x - spawn_offset, centroid.y};

  Rarity random_rarity = SampleRandomExpGemRarity(entity_config_);
  const ExpGemRarityConfig& gem_config =
      entity_config_.exp_gem.rarities[random_rarity];
  const Size2D gem_sprite_size = {gem_config.width, gem_config.height};
  pending_exp_gem_spawns_.push_back({random_rarity, gem_position, gem_position,
                                     CreateCenteredCollider(gem_sprite_size),
                                     entity_config_.exp_gem.inv_mass,
                                     gem_sprite_size});

  if (chest_will_spawn) {
    Vector2D chest_position = {centroid.x + spawn_offset, centroid.y};
    const Size2D chest_sprite_size = {entity_config_.chest.width,
                                      entity_config_.chest.height};
    pending_chest_spawns_.push_back({chest_position, chest_position,
                                     CreateCenteredCollider(chest_sprite_size),
                                     entity_config_.chest.inv_mass,
                                     chest_sprite_size});
  }

  pending_enemy_respawns_.push_back(event.enemy_idx);
}

void EntityManager::OnPlayerExpGemCollision(
    const PlayerExpGemCollisionEvent& event, EventContext& context) {
  const Rarity rarity = context.scene.exp_gem.rarity_[event.gem_idx];
  int exp_value = entity_config_.exp_gem.rarities[rarity].exp_value;
  context.scene.player.stats_.exp_points += exp_value;
  context.scene.exp_gem.to_be_destroyed_.insert(event.gem_idx);
  event_manager_->Emit(ExpGemCollectedEvent{event.gem_idx, exp_value});
}

void EntityManager::OnPlayerChestCollision(
    const PlayerChestCollisionEvent& event, EventContext& context) {
  context.scene.chest.to_be_destroyed_.insert(event.chest_idx);
  event_manager_->Emit(ChestOpenedEvent{event.chest_idx});
}

void EntityManager::OnPlayerDamaged(const PlayerDamagedEvent& event,
                                    EventContext& context) {
  if (!context.scene.player.is_alive_ || context.scene.player.is_invulnerable) {
    return;
  }

  context.scene.player.TakeDamage(event.damage);
  if (context.scene.player.is_alive_ &&
      context.scene.player.stats_.health <= 0) {
    context.scene.player.is_alive_ = false;
    event_manager_->Emit(PlayerDeadEvent{});
  }
}

void EntityManager::OnPlayerEnemyCollision(
    const PlayerEnemyCollisionEvent& event, EventContext& context) {
  int idx = event.enemy_idx;
  if (context.scene.enemy.attack_cooldown_timer[idx] < 0.0f) {
    int attack_damage = context.scene.enemy.attack_damage[idx];
    context.scene.enemy.damage_dealt_sim_step[idx] += attack_damage;
    context.scene.enemy.attack_cooldown_timer[idx] =
        context.scene.enemy.attack_cooldown_s[idx];
    event_manager_->Emit(PlayerDamagedEvent{idx, attack_damage});
  }
}

void EntityManager::OnEnemyDamaged(const EnemyDamagedEvent& event,
                                   EventContext& context) {
  int enemy_idx = event.enemy_idx;
  if (!context.scene.enemy.is_alive[enemy_idx]) {
    return;
  }

  context.scene.enemy.health_points[enemy_idx] -= event.damage;
  if (context.scene.enemy.health_points[enemy_idx] <= 0) {
    context.scene.enemy.is_alive[enemy_idx] = false;
    context.scene.enemy.is_done[enemy_idx] = true;
    context.scene.enemy.is_terminated_latched[enemy_idx] = true;
    event_manager_->Emit(EnemyKilledEvent{enemy_idx});
  }
}

void EntityManager::OnEnemyProjectileCollision(
    const EnemyProjectileCollisionEvent& event, EventContext& context) {
  context.scene.projectiles.to_be_destroyed_.insert(event.proj_idx);
  int proj_id = context.scene.projectiles.proj_type_[event.proj_idx];
  int spell_damage = context.scene.player.spell_stats_.damage[proj_id];
  int total_damage = context.scene.player.CalculateOutgoingDamage(spell_damage);
  event_manager_->Emit(EnemyDamagedEvent{event.enemy_idx, total_damage});
}

// ---------------------------------------------------------------------------
// Entity Lifecycle
// ---------------------------------------------------------------------------

void EntityManager::ProcessPendingSpawns(Scene& scene) {
  for (const ExpGemData& gem_data : pending_exp_gem_spawns_) {
    scene.exp_gem.AddExpGem(gem_data);
  }
  pending_exp_gem_spawns_.clear();

  for (const ChestData& chest_data : pending_chest_spawns_) {
    scene.chest.AddChest(chest_data);
  }
  pending_chest_spawns_.clear();

  for (const int enemy_idx : pending_enemy_respawns_) {
    RespawnEnemyAtIndex(scene.enemy, scene.player, enemy_idx);
  }
  pending_enemy_respawns_.clear();
}

void EntityManager::Cleanup(Scene& scene) {
  ResolveProjectileDestruction(scene);
  ResolveExpGemDestruction(scene);
  ResolveChestDestruction(scene);
}

void EntityManager::ResolveProjectileDestruction(Scene& scene) {
  for (int idx : scene.projectiles.to_be_destroyed_) {
    event_manager_->Emit(ProjectileDestroyedEvent{idx});
  }
  scene.projectiles.DestroyProjectiles();
}

void EntityManager::ResolveExpGemDestruction(Scene& scene) {
  scene.exp_gem.DestroyExpGems();
}

void EntityManager::ResolveChestDestruction(Scene& scene) {
  scene.chest.DestroyChests();
}

}  // namespace arelto
