// src/entity_manager.cpp
#include "entity_manager.h"
#include "constants/chest.h"
#include "constants/enemy.h"
#include "constants/exp_gem.h"
#include "event_manager.h"
#include "random.h"
#include "types.h"

namespace arelto {

EntityManager::EntityManager() {}
EntityManager::~EntityManager() {}

void EntityManager::Initialize(EventManager& event_manager) {
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

  event_manager.Subscribe<PlayerHealedEvent>(
      [this](const PlayerHealedEvent& event, EventContext& event_context) {
        OnPlayerHealed(event, event_context);
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

void EntityManager::OnEnemyKilled(const EnemyKilledEvent& event,
                                  EventContext& context) {
  Vector2D centroid =
      GetCentroid(context.scene.enemy.position[event.enemy_idx],
                  context.scene.enemy.collider[event.enemy_idx].size);

  float chest_roll = static_cast<float>(GenerateRandomInt(0, 99)) / 100.0f;
  bool chest_will_spawn = chest_roll < kChestSpawnChance;

  float spawn_offset = chest_will_spawn ? kGemChestMinSeparation * 0.5f : 0.0f;
  Vector2D gem_position = {centroid.x - spawn_offset, centroid.y};

  Rarity random_rarity = static_cast<Rarity>(GenerateRandomInt(0, 3));
  pending_exp_gem_spawns_.push_back({random_rarity, gem_position, gem_position,
                                     kExpGemCollider[random_rarity],
                                     kExpGemInvMass,
                                     kExpGemSpriteSize[random_rarity]});

  if (chest_will_spawn) {
    Vector2D chest_position = {centroid.x + spawn_offset, centroid.y};
    pending_chest_spawns_.push_back({chest_position, chest_position,
                                     kChestCollider, kChestInvMass,
                                     kChestSpriteSize});
  }

  pending_enemy_respawns_.push_back(event.enemy_idx);
}

void EntityManager::OnPlayerExpGemCollision(
    const PlayerExpGemCollisionEvent& event, EventContext& context) {
  int exp_value = kExpGemValues[context.scene.exp_gem.rarity_[event.gem_idx]];
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

void EntityManager::OnPlayerHealed(const PlayerHealedEvent& event,
                                   EventContext& context) {
  context.scene.player.TakeHealing(event.healing);
}

void EntityManager::OnPlayerEnemyCollision(
    const PlayerEnemyCollisionEvent& event, EventContext& context) {
  int idx = event.enemy_idx;
  if (context.scene.enemy.attack_cooldown[idx] < 0.0f) {
    int attack_damage = context.scene.enemy.attack_damage[idx];
    context.scene.enemy.damage_dealt_sim_step[idx] += attack_damage;
    context.scene.enemy.attack_cooldown[idx] = kEnemyAttackCooldown;
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
  event_manager_->Emit(EnemyDamagedEvent{event.enemy_idx, spell_damage});
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
