// src/entity_manager.cpp
#include "entity_manager.h"
#include <algorithm>
#include "constants/chest.h"
#include "constants/enemy.h"
#include "constants/exp_gem.h"
#include "event_manager.h"
#include "random.h"
#include "types.h"

namespace arelto {

EntityManager::EntityManager() {}
EntityManager::~EntityManager() {}

void EntityManager::Initialize(Scene& scene, EventManager& event_manager) {
  // Scene and event manager references are stored for use in event handlers and lifecycle methods.
  // This simplifies the function signatures of those methods and allows them to be used as event
  // handlers without needing to bind extra arguments or use long, verbose lambdas.
  scene_ = &scene;
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

  event_manager.Subscribe<PlayerEnemyCollisionEvent>(
      [this](const PlayerEnemyCollisionEvent& event,
             EventContext& event_context) {
        OnPlayerEnemyCollision(event, event_context);
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
                                  EventContext& /*context*/) {
  Vector2D centroid = GetCentroid(scene_->enemy.position[event.enemy_idx],
                                  scene_->enemy.collider[event.enemy_idx].size);

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
    const PlayerExpGemCollisionEvent& event, EventContext& /*event_context*/) {
  int exp_value = kExpGemValues[scene_->exp_gem.rarity_[event.gem_idx]];
  scene_->player.stats_.exp_points += exp_value;
  scene_->exp_gem.to_be_destroyed_.insert(event.gem_idx);
  event_manager_->Emit(ExpGemCollectedEvent{event.gem_idx, exp_value});
}

void EntityManager::OnPlayerChestCollision(
    const PlayerChestCollisionEvent& event, EventContext& /*context*/) {
  scene_->chest.to_be_destroyed_.insert(event.chest_idx);
  event_manager_->Emit(ChestOpenedEvent{event.chest_idx});
}

void EntityManager::OnPlayerEnemyCollision(
    const PlayerEnemyCollisionEvent& event, EventContext& /*event_context*/) {
  int idx = event.enemy_idx;

  if (scene_->enemy.attack_cooldown[idx] < 0.0f) {
    int damage_dealt = std::max(
        0, scene_->enemy.attack_damage[idx] -
               static_cast<int>(scene_->player.stats_.armor.GetValue()));
    scene_->player.stats_.health -= damage_dealt;
    scene_->enemy.damage_dealt_sim_step[idx] +=
        scene_->enemy.attack_damage[idx];
    scene_->enemy.attack_cooldown[idx] = kEnemyAttackCooldown;
    event_manager_->Emit(PlayerDamagedEvent{idx, damage_dealt});

    if (scene_->player.is_alive_ && scene_->player.stats_.health <= 0) {
      scene_->player.is_alive_ = false;
      event_manager_->Emit(PlayerDeadEvent{});
    }
  }
}

void EntityManager::OnEnemyProjectileCollision(
    const EnemyProjectileCollisionEvent& event,
    EventContext& /*event_context*/) {
  scene_->projectiles.to_be_destroyed_.insert(event.proj_idx);
  int proj_id = scene_->projectiles.proj_type_[event.proj_idx];
  int spell_damage = scene_->player.spell_stats_.damage[proj_id];
  scene_->enemy.health_points[event.enemy_idx] -= spell_damage;
  int enemy_idx = event.enemy_idx;
  if (scene_->enemy.is_alive[enemy_idx] &&
      scene_->enemy.health_points[enemy_idx] <= 0) {
    scene_->enemy.is_alive[enemy_idx] = false;
    scene_->enemy.is_done[enemy_idx] = true;
    scene_->enemy.is_terminated_latched[enemy_idx] = true;
    event_manager_->Emit(EnemyKilledEvent{enemy_idx});
  };
}

// ---------------------------------------------------------------------------
// Entity Lifecycle
// ---------------------------------------------------------------------------

void EntityManager::ProcessPendingSpawns() {
  for (const ExpGemData& gem_data : pending_exp_gem_spawns_) {
    scene_->exp_gem.AddExpGem(gem_data);
  }
  pending_exp_gem_spawns_.clear();

  for (const ChestData& chest_data : pending_chest_spawns_) {
    scene_->chest.AddChest(chest_data);
  }
  pending_chest_spawns_.clear();

  for (const int enemy_idx : pending_enemy_respawns_) {
    RespawnEnemyAtIndex(scene_->enemy, scene_->player, enemy_idx);
  }
  pending_enemy_respawns_.clear();
}

void EntityManager::Cleanup() {
  ResolveProjectileDestruction();
  ResolveExpGemDestruction();
  ResolveChestDestruction();
}

void EntityManager::ResolveProjectileDestruction() {
  for (int idx : scene_->projectiles.to_be_destroyed_) {
    event_manager_->Emit(ProjectileDestroyedEvent{idx});
  }
  scene_->projectiles.DestroyProjectiles();
}

void EntityManager::ResolveExpGemDestruction() {
  scene_->exp_gem.DestroyExpGems();
}

void EntityManager::ResolveChestDestruction() {
  scene_->chest.DestroyChests();
}

}  // namespace arelto
