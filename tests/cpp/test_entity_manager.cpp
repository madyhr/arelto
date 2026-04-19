// tests/cpp/test_entity_manager.cpp
// Unit tests for EntityManager class

#include <gtest/gtest.h>

#include "constants/enemy.h"
#include "entity_manager.h"
#include "event_manager.h"
#include "scene.h"
#include "test_helpers.h"

namespace arelto {
namespace {

class EntityManagerTest : public ::testing::Test {
 protected:
  void SetUp() override { scene_ = testing::CreateTestScene(); }

  Scene scene_;
  EntityManager entity_manager_;
  EventManager event_manager_;
};

// =============================================================================
// Cleanup - Projectile Tests
// =============================================================================

TEST_F(EntityManagerTest, Cleanup_ProjectileMarkedForDestruction_IsDestroyed) {
  entity_manager_.Initialize(event_manager_);
  // Add a projectile
  ProjectileData proj = testing::CreateProjectileAt(100.0f, 100.0f, 1.0f, 0.0f);
  scene_.projectiles.AddProjectile(proj);
  ASSERT_EQ(scene_.projectiles.GetNumProjectiles(), 1);

  // Mark it for destruction
  scene_.projectiles.to_be_destroyed_.insert(0);
  entity_manager_.Cleanup(scene_);

  // Projectile should be removed
  EXPECT_EQ(scene_.projectiles.GetNumProjectiles(), 0);
}

TEST_F(EntityManagerTest, Cleanup_NoMarkedProjectiles_CountUnchanged) {
  entity_manager_.Initialize(event_manager_);
  // Add projectiles
  for (int i = 0; i < 3; ++i) {
    ProjectileData proj = testing::CreateProjectileAt(
        100.0f + static_cast<float>(i) * 50.0f, 100.0f, 1.0f, 0.0f);
    scene_.projectiles.AddProjectile(proj);
  }
  ASSERT_EQ(scene_.projectiles.GetNumProjectiles(), 3);
  entity_manager_.Cleanup(scene_);

  // All projectiles should still exist
  EXPECT_EQ(scene_.projectiles.GetNumProjectiles(), 3);
}

// =============================================================================
// Cleanup - Gem Tests
// =============================================================================

TEST_F(EntityManagerTest, Cleanup_GemMarkedForDestruction_IsDestroyed) {
  entity_manager_.Initialize(event_manager_);
  // Add a gem
  ExpGemData gem_data{Rarity::common,
                      {100.0f, 100.0f},
                      {100.0f, 100.0f},
                      {{8.0f, 8.0f}, {16, 16}},
                      5.0f,
                      {16, 16}};
  scene_.exp_gem.AddExpGem(gem_data);
  ASSERT_EQ(scene_.exp_gem.GetNumExpGems(), 1);

  // Mark it for destruction
  scene_.exp_gem.to_be_destroyed_.insert(0);
  entity_manager_.Cleanup(scene_);

  // Gem should be removed
  EXPECT_EQ(scene_.exp_gem.GetNumExpGems(), 0);
}

// =============================================================================
// Event Emission Tests
// =============================================================================

TEST_F(EntityManagerTest,
       CleanupProjectilesStatus_EmitsProjectileDestroyedEvent) {
  ProjectileData proj = testing::CreateProjectileAt(100.0f, 100.0f, 1.0f, 0.0f);
  entity_manager_.Initialize(event_manager_);
  scene_.projectiles.AddProjectile(proj);
  scene_.projectiles.AddProjectile(proj);

  // Mark both for destruction
  scene_.projectiles.to_be_destroyed_.insert(0);
  scene_.projectiles.to_be_destroyed_.insert(1);
  entity_manager_.Cleanup(scene_);

  int destroyed_count = 0;
  for (const auto& e : event_manager_.GetEvents()) {
    if (std::holds_alternative<ProjectileDestroyedEvent>(e)) {
      ++destroyed_count;
    }
  }
  EXPECT_EQ(destroyed_count, 2);
}

// =============================================================================
// ProcessPendingSpawns - Enemy Respawn Tests
// =============================================================================

TEST_F(EntityManagerTest,
       ProcessPendingSpawns_EnemyKilledEventEmitted_EnemyRespawns) {
  entity_manager_.Initialize(event_manager_);

  // Kill enemy at index 0
  int enemy_idx = 0;
  scene_.enemy.is_alive[enemy_idx] = false;
  scene_.enemy.is_done[enemy_idx] = true;
  scene_.enemy.health_points[enemy_idx] = 0;

  // Emit the kill event and dispatch it so OnEnemyKilled queues the respawn
  event_manager_.Emit(EnemyKilledEvent{enemy_idx});
  EventContext event_context = testing::MakeEventContext(scene_, event_manager_);
  event_manager_.Dispatch(event_context);

  entity_manager_.ProcessPendingSpawns(scene_);

  EXPECT_TRUE(scene_.enemy.is_alive[enemy_idx]);
  EXPECT_FALSE(scene_.enemy.is_done[enemy_idx]);
  EXPECT_EQ(scene_.enemy.health_points[enemy_idx], kEnemyHealth);
}

}  // namespace
}  // namespace arelto
