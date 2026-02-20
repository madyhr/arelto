// tests/cpp/test_entity_manager.cpp
// Unit tests for EntityManager class

#include <gtest/gtest.h>

#include "constants/enemy.h"
#include "constants/ray_caster.h"
#include "entity_manager.h"
#include "scene.h"
#include "test_helpers.h"

namespace arelto {
namespace {

class EntityManagerTest : public ::testing::Test {
 protected:
  void SetUp() override { scene_ = testing::CreateTestScene(); }

  Scene scene_;
  EntityManager entity_manager_;
};

// =============================================================================
// IsPlayerDead Tests
// =============================================================================

TEST_F(EntityManagerTest, IsPlayerDead_WorksCorrectly) {
  // Dead cases
  scene_.player.stats_.health = 0;
  EXPECT_TRUE(entity_manager_.IsPlayerDead(scene_.player));

  scene_.player.stats_.health = -10;
  EXPECT_TRUE(entity_manager_.IsPlayerDead(scene_.player));

  // Alive cases
  scene_.player.stats_.health = 100;
  EXPECT_FALSE(entity_manager_.IsPlayerDead(scene_.player));

  scene_.player.stats_.health = 1;
  EXPECT_FALSE(entity_manager_.IsPlayerDead(scene_.player));
}

// =============================================================================
// Update - Enemy Status Tests
// =============================================================================

TEST_F(EntityManagerTest, Update_EnemyStatus_TransitionsCorrectly) {
  // Test Alive -> Alive
  scene_.enemy.health_points[0] = 100;
  scene_.enemy.is_alive[0] = true;
  scene_.enemy.is_done[0] = false;

  entity_manager_.Update(scene_, 0.016f);
  EXPECT_TRUE(scene_.enemy.is_alive[0]);
  EXPECT_FALSE(scene_.enemy.is_done[0]);

  // Test Alive -> Dead (Zero Health)
  scene_.enemy.health_points[0] = 0;
  entity_manager_.Update(scene_, 0.016f);
  EXPECT_FALSE(scene_.enemy.is_alive[0]);
  EXPECT_TRUE(scene_.enemy.is_done[0]);

  // Test Alive -> Dead (Negative Health)
  scene_.enemy.health_points[1] = -5;
  scene_.enemy.is_alive[1] = true;
  scene_.enemy.is_done[1] = false;

  entity_manager_.Update(scene_, 0.016f);
  EXPECT_FALSE(scene_.enemy.is_alive[1]);
  EXPECT_TRUE(scene_.enemy.is_done[1]);
}

TEST_F(EntityManagerTest, Update_EnemyTimeoutTimerIncreases) {
  float initial_timer = scene_.enemy.timeout_timer[0];

  entity_manager_.Update(scene_, 0.016f);

  EXPECT_GT(scene_.enemy.timeout_timer[0], initial_timer);
}

TEST_F(EntityManagerTest, Update_DeadEnemy_SetsTerminatedLatched) {
  scene_.enemy.health_points[0] = 0;
  scene_.enemy.is_alive[0] = true;
  scene_.enemy.is_terminated_latched[0] = false;

  entity_manager_.Update(scene_, 0.016f);

  EXPECT_TRUE(scene_.enemy.is_terminated_latched[0]);
}

// =============================================================================
// Update - Projectile Status Tests
// =============================================================================

TEST_F(EntityManagerTest, Update_ProjectileMarkedForDestruction_IsDestroyed) {
  // Add a projectile
  ProjectileData proj = testing::CreateProjectileAt(100.0f, 100.0f, 1.0f, 0.0f);
  scene_.projectiles.AddProjectile(proj);
  ASSERT_EQ(scene_.projectiles.GetNumProjectiles(), 1);

  // Mark it for destruction
  scene_.projectiles.to_be_destroyed_.insert(0);

  entity_manager_.Update(scene_, 0.016f);

  // Projectile should be removed
  EXPECT_EQ(scene_.projectiles.GetNumProjectiles(), 0);
}

TEST_F(EntityManagerTest, Update_NoMarkedProjectiles_CountUnchanged) {
  // Add projectiles
  for (int i = 0; i < 3; ++i) {
    ProjectileData proj =
        testing::CreateProjectileAt(100.0f + i * 50.0f, 100.0f, 1.0f, 0.0f);
    scene_.projectiles.AddProjectile(proj);
  }
  ASSERT_EQ(scene_.projectiles.GetNumProjectiles(), 3);

  entity_manager_.Update(scene_, 0.016f);

  // All projectiles should still exist
  EXPECT_EQ(scene_.projectiles.GetNumProjectiles(), 3);
}

// =============================================================================
// Update - Gem Status Tests
// =============================================================================

TEST_F(EntityManagerTest, Update_GemMarkedForDestruction_IsDestroyed) {
  // Add a gem
  ExpGemData gem_data{Rarity::common,
                      {100.0f, 100.0f},
                      {100.0f, 100.0f},
                      {{8.0f, 8.0f}, {16, 16}},
                      {16, 16}};
  scene_.exp_gem.AddExpGem(gem_data);
  ASSERT_EQ(scene_.exp_gem.GetNumExpGems(), 1);

  // Mark it for destruction
  scene_.exp_gem.to_be_destroyed_.insert(0);

  entity_manager_.Update(scene_, 0.016f);

  // Gem should be removed
  EXPECT_EQ(scene_.exp_gem.GetNumExpGems(), 0);
}

// =============================================================================
// Update - Multiple Entities Tests
// =============================================================================

TEST_F(EntityManagerTest, Update_MultipleDeadEnemies_AllMarkedCorrectly) {
  // Kill multiple enemies
  scene_.enemy.health_points[0] = 0;
  scene_.enemy.health_points[2] = -5;
  scene_.enemy.health_points[4] = 0;

  for (int i = 0; i < kNumEnemies; ++i) {
    scene_.enemy.is_alive[i] = true;
    scene_.enemy.is_done[i] = false;
  }

  entity_manager_.Update(scene_, 0.016f);

  // Check dead enemies
  EXPECT_FALSE(scene_.enemy.is_alive[0]);
  EXPECT_TRUE(scene_.enemy.is_done[0]);

  EXPECT_FALSE(scene_.enemy.is_alive[2]);
  EXPECT_TRUE(scene_.enemy.is_done[2]);

  EXPECT_FALSE(scene_.enemy.is_alive[4]);
  EXPECT_TRUE(scene_.enemy.is_done[4]);

  // Check alive enemies
  EXPECT_TRUE(scene_.enemy.is_alive[1]);
  EXPECT_FALSE(scene_.enemy.is_done[1]);

  EXPECT_TRUE(scene_.enemy.is_alive[3]);
  EXPECT_FALSE(scene_.enemy.is_done[3]);
}

// =============================================================================
// Update - Integration Tests
// =============================================================================

TEST_F(EntityManagerTest, Update_UpdatesRayCaster) {
  // Place an enemy and player nearby
  scene_.player.position_ = {100.0f, 100.0f};
  scene_.enemy.position[0] = {
      250.0f,
      100.0f};  // Further away to ensure ray start is outside player grid cell
  scene_.enemy.is_alive[0] = true;

  // Clear any existing ray data
  int history_idx = scene_.enemy.ray_caster.history_idx;
  for (int r = 0; r < kNumRays; ++r) {
    scene_.enemy.ray_caster.ray_hit_distances[history_idx][r][0] = 0.0f;
  }

  entity_manager_.Update(scene_, 0.016f);

  // Check if ray caster data was updated
  // We expect some non-zero distances since player is nearby
  bool found_hit = false;
  int new_history_idx = scene_.enemy.ray_caster.history_idx;
  int checked_idx =
      (new_history_idx - 1 + kRayHistoryLength) % kRayHistoryLength;

  for (int r = 0; r < kNumRays; ++r) {
    if (scene_.enemy.ray_caster.ray_hit_distances[checked_idx][r][0] > 0.0f) {
      found_hit = true;
      break;
    }
  }
  EXPECT_TRUE(found_hit)
      << "Ray caster did not detect the nearby player after update";
}

TEST_F(EntityManagerTest, Update_UpdatesRayCaster_DetectsProjectiles) {
  // Place an enemy and projectile nearby
  scene_.enemy.position[0] = {100.0f, 100.0f};
  scene_.enemy.is_alive[0] = true;

  // Create a projectile to the right of the enemy
  // Note: Must be placed outside the ray start offset radius
  ProjectileData proj = testing::CreateProjectileAt(200.0f, 100.0f, 1.0f, 0.0f);
  scene_.projectiles.AddProjectile(proj);

  // Clear any existing ray data
  int history_idx = scene_.enemy.ray_caster.history_idx;
  for (int r = 0; r < kNumRays; ++r) {
    scene_.enemy.ray_caster.non_blocking_ray_hit_distances[history_idx][r][0] =
        0.0f;
  }

  entity_manager_.Update(scene_, 0.016f);

  // Check if ray caster data was updated for projectiles
  bool found_hit = false;
  int new_history_idx = scene_.enemy.ray_caster.history_idx;
  int checked_idx =
      (new_history_idx - 1 + kRayHistoryLength) % kRayHistoryLength;

  for (int r = 0; r < kNumRays; ++r) {
    if (scene_.enemy.ray_caster
            .non_blocking_ray_hit_distances[checked_idx][r][0] > 0.0f) {
      found_hit = true;
      break;
    }
  }
  EXPECT_TRUE(found_hit)
      << "Ray caster did not detect the nearby projectile after update";
}

}  // namespace
}  // namespace arelto
