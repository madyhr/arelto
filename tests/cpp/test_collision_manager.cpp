// tests/cpp/test_collision_manager.cpp
// Unit tests for CollisionManager class

#include <gtest/gtest.h>

#include <array>

#include "collision_manager.h"
#include "constants/enemy.h"
#include "entity.h"
#include "scene.h"
#include "test_helpers.h"
#include "types.h"

namespace arelto {
namespace {

class CollisionManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = testing::CreateTestScene();
    testing::DeactivateAllEnemies(scene_.enemy);
  }

  Scene scene_;
  CollisionManager collision_manager_;
};

// Note: GetCollisionType and GetDisplacementVectors are private methods.
// We test their behavior indirectly through HandleCollisionsSAP.

// =============================================================================
// HandleCollisionsSAP Integration Tests
// =============================================================================

TEST_F(CollisionManagerTest, HandleCollisionsSAP_NoCollisions_NoChanges) {
  Vector2D initial_player_pos = scene_.player.position_;
  std::array<Vector2D, kNumEnemies> initial_enemy_pos;
  for (int i = 0; i < kNumEnemies; ++i) {
    initial_enemy_pos[i] = scene_.enemy.position[i];
  }

  collision_manager_.HandleCollisionsSAP(scene_);

  // No positions should have changed
  testing::ExpectVector2DEq(scene_.player.position_, initial_player_pos);
  for (int i = 0; i < kNumEnemies; ++i) {
    testing::ExpectVector2DEq(scene_.enemy.position[i], initial_enemy_pos[i]);
  }
}

TEST_F(CollisionManagerTest,
       HandleCollisionsSAP_PlayerEnemyCollision_SeparatesEntities) {
  // Place player and first enemy overlapping
  scene_.player.position_ = {100.0f, 100.0f};
  scene_.enemy.position[0] = {100.0f, 100.0f};  // Exact same position
  scene_.enemy.is_alive[0] = true;

  collision_manager_.HandleCollisionsSAP(scene_);

  // Player and enemy should be separated (positions should differ)
  float distance = (scene_.player.position_ - scene_.enemy.position[0]).Norm();
  // They should have been pushed apart
  EXPECT_GT(distance, 0.0f);
}

TEST_F(CollisionManagerTest,
       HandleCollisionsSAP_EnemyEnemyCollision_SeparatesEntities) {
  // Place two enemies overlapping
  scene_.enemy.position[0] = {100.0f, 100.0f};
  scene_.enemy.position[1] = {100.0f, 100.0f};
  scene_.enemy.is_alive[0] = true;
  scene_.enemy.is_alive[1] = true;

  scene_.player.position_ = {10000.0f, 10000.0f};

  collision_manager_.HandleCollisionsSAP(scene_);

  // Enemies should be separated
  float distance = (scene_.enemy.position[0] - scene_.enemy.position[1]).Norm();
  EXPECT_GT(distance, 0.0f);
}

TEST_F(CollisionManagerTest, HandleCollisionsSAP_DeadEnemyIgnored) {
  // Place player overlapping with a dead enemy
  scene_.player.position_ = {100.0f, 100.0f};
  scene_.enemy.position[0] = {100.0f, 100.0f};
  scene_.enemy.is_alive[0] = false;  // Dead enemy

  Vector2D initial_player_pos = scene_.player.position_;

  collision_manager_.HandleCollisionsSAP(scene_);

  // Player should not have moved (dead enemies are ignored)
  testing::ExpectVector2DEq(scene_.player.position_, initial_player_pos);
}

TEST_F(CollisionManagerTest,
       HandleCollisionsSAP_PlayerEnemyCollision_DealsDamage) {
  // Place player and enemy overlapping
  scene_.player.position_ = {100.0f, 100.0f};
  scene_.player.stats_.health = 100;
  scene_.enemy.position[0] = {100.0f, 100.0f};
  scene_.enemy.is_alive[0] = true;
  scene_.enemy.attack_cooldown[0] = -1.0f;  // Ready to attack
  scene_.enemy.attack_damage[0] = 10;

  collision_manager_.HandleCollisionsSAP(scene_);

  // Player should have taken damage
  EXPECT_EQ(scene_.player.stats_.health, 90);
}

TEST_F(CollisionManagerTest,
       HandleCollisionsSAP_PlayerEnemyCollision_RespectsAttackCooldown) {
  // Place player and enemy overlapping
  scene_.player.position_ = {100.0f, 100.0f};
  scene_.player.stats_.health = 100;
  scene_.enemy.position[0] = {100.0f, 100.0f};
  scene_.enemy.is_alive[0] = true;
  scene_.enemy.attack_cooldown[0] = 1.0f;  // On cooldown
  scene_.enemy.attack_damage[0] = 10;

  collision_manager_.HandleCollisionsSAP(scene_);

  // Player should NOT have taken damage (enemy on cooldown)
  EXPECT_EQ(scene_.player.stats_.health, 100);
}

TEST_F(CollisionManagerTest,
       HandleCollisionsSAP_PlayerGemCollision_CollectsExp) {
  // Place player overlapping with a gem
  scene_.player.position_ = {100.0f, 100.0f};
  scene_.player.stats_.exp_points = 0;

  // Add a gem at player's position
  ExpGemData gem_data{Rarity::common,
                      {100.0f, 100.0f},
                      {100.0f, 100.0f},
                      {{8.0f, 8.0f}, {16, 16}},
                      {16, 16}};
  scene_.exp_gem.AddExpGem(gem_data);

  collision_manager_.HandleCollisionsSAP(scene_);

  // Gem should be marked for destruction
  EXPECT_TRUE(scene_.exp_gem.to_be_destroyed_.count(0) > 0);
  // Player should have gained exp
  EXPECT_GT(scene_.player.stats_.exp_points, 0);
}

TEST_F(CollisionManagerTest,
       HandleCollisionsSAP_EnemyProjectileCollision_DealsDamage) {
  // Place enemy and projectile overlapping
  scene_.enemy.position[0] = {100.0f, 100.0f};
  scene_.enemy.is_alive[0] = true;
  scene_.enemy.health_points[0] = 100;

  // Add a projectile at enemy's position
  ProjectileData proj = testing::CreateProjectileAt(100.0f, 100.0f, 1.0f, 0.0f);
  scene_.projectiles.AddProjectile(proj);

  // Set player spell damage
  scene_.player.spell_stats_.damage[0] = 25;

  // Move player away
  scene_.player.position_ = {10000.0f, 10000.0f};

  collision_manager_.HandleCollisionsSAP(scene_);

  // Projectile should be marked for destruction
  EXPECT_TRUE(scene_.projectiles.to_be_destroyed_.count(0) > 0);
  // Enemy should have taken damage
  EXPECT_EQ(scene_.enemy.health_points[0], 75);
}

}  // namespace
}  // namespace arelto
