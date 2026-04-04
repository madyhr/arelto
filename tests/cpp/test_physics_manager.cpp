// tests/cpp/test_physics_manager.cpp
// Unit tests for PhysicsManager class

#include <gtest/gtest.h>

#include "constants/map.h"
#include "event_manager.h"
#include "physics_manager.h"
#include "scene.h"
#include "test_helpers.h"

namespace arelto {
namespace {

class PhysicsManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = testing::CreateTestScene();
    // Ensure accurate testing by starting with no active enemies (unless test specifically activates them)
    // Randomly spawned enemies can cause unwanted collisions/behavior in tests assuming isolation.
    testing::DeactivateAllEnemies(scene_.enemy);
    physics_manager_.Initialize();
  }

  Scene scene_;
  PhysicsManager physics_manager_;
  EventManager event_manager_;
};

// =============================================================================
// StepPhysics - Player Movement Tests
// =============================================================================

TEST_F(PhysicsManagerTest, StepPhysics_PlayerMovesWithPositiveVelocity) {
  scene_.player.position_ = {100.0f, 100.0f};
  scene_.player.velocity_ = {1.0f, 0.0f};  // Moving right

  Vector2D initial_pos = scene_.player.position_;
  physics_manager_.StepPhysics(scene_, event_manager_);

  // Player should have moved right
  EXPECT_GT(scene_.player.position_.x, initial_pos.x);
  EXPECT_FLOAT_EQ(scene_.player.position_.y, initial_pos.y);
}

TEST_F(PhysicsManagerTest, StepPhysics_PlayerMovesWithNegativeVelocity) {
  scene_.player.position_ = {100.0f, 100.0f};
  scene_.player.velocity_ = {-1.0f, -1.0f};  // Moving up-left

  Vector2D initial_pos = scene_.player.position_;
  physics_manager_.StepPhysics(scene_, event_manager_);

  // Player should have moved up-left
  EXPECT_LT(scene_.player.position_.x, initial_pos.x);
  EXPECT_LT(scene_.player.position_.y, initial_pos.y);
}

TEST_F(PhysicsManagerTest, StepPhysics_PlayerTracksLastHorizontalVelocity) {
  scene_.player.position_ = {100.0f, 100.0f};
  scene_.player.velocity_ = {1.0f, 0.0f};
  scene_.player.last_horizontal_velocity_ = 0.0f;

  physics_manager_.StepPhysics(scene_, event_manager_);

  EXPECT_GT(scene_.player.last_horizontal_velocity_, 0.0f);
}

// =============================================================================
// StepPhysics - Enemy Movement Tests
// =============================================================================

TEST_F(PhysicsManagerTest, StepPhysics_EnemyMovesWithVelocity) {
  scene_.enemy.position[0] = {100.0f, 100.0f};
  scene_.enemy.velocity[0] = {1.0f, 0.0f};
  scene_.enemy.is_alive[0] = true;

  Vector2D initial_pos = scene_.enemy.position[0];
  physics_manager_.StepPhysics(scene_, event_manager_);

  // Enemy should have moved
  EXPECT_GT(scene_.enemy.position[0].x, initial_pos.x);
}

TEST_F(PhysicsManagerTest, StepPhysics_DeadEnemyDoesNotMove) {
  scene_.enemy.position[0] = {100.0f, 100.0f};
  scene_.enemy.velocity[0] = {1.0f, 0.0f};
  scene_.enemy.is_alive[0] = false;  // Dead

  Vector2D initial_pos = scene_.enemy.position[0];
  physics_manager_.StepPhysics(scene_, event_manager_);

  // Dead enemy should not have moved
  testing::ExpectVector2DEq(scene_.enemy.position[0], initial_pos);
}

TEST_F(PhysicsManagerTest, StepPhysics_EnemyVelocityNormalized) {
  scene_.enemy.position[0] = {100.0f, 100.0f};
  scene_.enemy.velocity[0] = {3.0f, 4.0f};  // Non-normalized (norm = 5)
  scene_.enemy.is_alive[0] = true;

  physics_manager_.StepPhysics(scene_, event_manager_);

  // Velocity should be normalized after step
  float norm = scene_.enemy.velocity[0].Norm();
  EXPECT_NEAR(norm, 1.0f, 1e-5f);
}

TEST_F(PhysicsManagerTest, StepPhysics_EnemyAttackCooldownDecreases) {
  scene_.enemy.is_alive[0] = true;
  scene_.enemy.attack_cooldown[0] = 1.0f;

  physics_manager_.StepPhysics(scene_, event_manager_);

  // Cooldown should have decreased
  EXPECT_LT(scene_.enemy.attack_cooldown[0], 1.0f);
}

TEST_F(PhysicsManagerTest, StepPhysics_EnemyDamageDealtResets) {
  scene_.enemy.is_alive[0] = true;
  scene_.enemy.damage_dealt_sim_step[0] = 10;

  physics_manager_.StepPhysics(scene_, event_manager_);

  // Damage dealt should be reset to 0
  EXPECT_EQ(scene_.enemy.damage_dealt_sim_step[0], 0);
}

// =============================================================================
// StepPhysics - Projectile Movement Tests
// =============================================================================

TEST_F(PhysicsManagerTest, StepPhysics_ProjectileMovesWithDirection) {
  ProjectileData proj = testing::CreateProjectileAt(100.0f, 100.0f, 1.0f, 0.0f);
  scene_.projectiles.AddProjectile(proj);

  Vector2D initial_pos = scene_.projectiles.position_[0];
  physics_manager_.StepPhysics(scene_, event_manager_);

  // Projectile should have moved right
  EXPECT_GT(scene_.projectiles.position_[0].x, initial_pos.x);
}

TEST_F(PhysicsManagerTest, StepPhysics_NoProjectiles_NoError) {
  // Clear all projectiles
  scene_.projectiles.ResetAllProjectiles();

  // Should not crash
  EXPECT_NO_THROW(physics_manager_.StepPhysics(scene_, event_manager_));
}

// =============================================================================
// Out of Bounds Tests
// =============================================================================

TEST_F(PhysicsManagerTest, StepPhysics_PlayerClampsToMapBounds) {
  // Left Bound
  scene_.player.position_ = {-100.0f, 100.0f};
  scene_.player.velocity_ = {0.0f, 0.0f};
  physics_manager_.StepPhysics(scene_, event_manager_);
  EXPECT_GE(scene_.player.position_.x, 0.0f);

  // Top Bound
  scene_.player.position_ = {100.0f, -100.0f};
  physics_manager_.StepPhysics(scene_, event_manager_);
  EXPECT_GE(scene_.player.position_.y, 0.0f);

  // Right Bound
  scene_.player.position_ = {static_cast<float>(kMapWidth) + 100.0f, 100.0f};
  physics_manager_.StepPhysics(scene_, event_manager_);
  EXPECT_LE(scene_.player.position_.x + scene_.player.stats_.sprite_size.width,
            static_cast<float>(kMapWidth));

  // Bottom Bound
  scene_.player.position_ = {100.0f, static_cast<float>(kMapHeight) + 100.0f};
  physics_manager_.StepPhysics(scene_, event_manager_);
  EXPECT_LE(scene_.player.position_.y + scene_.player.stats_.sprite_size.height,
            static_cast<float>(kMapHeight));
}

TEST_F(PhysicsManagerTest, StepPhysics_EnemyClampsToMapBounds) {
  scene_.enemy.position[0] = {-100.0f, -100.0f};
  scene_.enemy.is_alive[0] = true;
  scene_.enemy.velocity[0] = {0.0f, 0.0f};

  physics_manager_.StepPhysics(scene_, event_manager_);

  EXPECT_GE(scene_.enemy.position[0].x, 0.0f);
  EXPECT_GE(scene_.enemy.position[0].y, 0.0f);
}

TEST_F(PhysicsManagerTest, StepPhysics_ProjectileOOB_MarkedForDestruction) {
  // Add projectile at edge moving out of bounds
  ProjectileData proj =
      testing::CreateProjectileAt(-10.0f, 100.0f, -1.0f, 0.0f);
  scene_.projectiles.AddProjectile(proj);

  physics_manager_.StepPhysics(scene_, event_manager_);

  // Projectile should be marked for destruction
  EXPECT_TRUE(scene_.projectiles.to_be_destroyed_.count(0) > 0);
}

// =============================================================================
// Collision Resolution Causing Movement Tests
// =============================================================================

TEST_F(PhysicsManagerTest, StepPhysics_ResolvesCollisions) {
  // Place player and enemy in overlapping positions
  scene_.player.position_ = {100.0f, 100.0f};
  scene_.enemy.position[0] = {100.0f, 100.0f};
  scene_.enemy.is_alive[0] = true;

  // Set velocities to zero (so only collision resolution should move them)
  scene_.player.velocity_ = {0.0f, 0.0f};
  scene_.enemy.velocity[0] = {0.0f, 0.0f};

  Vector2D initial_player = scene_.player.position_;
  Vector2D initial_enemy = scene_.enemy.position[0];

  physics_manager_.StepPhysics(scene_, event_manager_);

  // They should have moved apart
  EXPECT_NE(scene_.player.position_, initial_player);
  EXPECT_NE(scene_.enemy.position[0], initial_enemy);

  float distance = (scene_.player.position_ - scene_.enemy.position[0]).Norm();
  EXPECT_GT(distance, 0.0f);
}

// =============================================================================
// Event Dispatch / Flush Integration Tests
// =============================================================================

TEST_F(PhysicsManagerTest, StepPhysics_EmitsEventsIntoQueue_ReadyForDispatch) {
  // Set up player-enemy collision
  scene_.player.position_ = {100.0f, 100.0f};
  scene_.enemy.position[0] = {100.0f, 100.0f};
  scene_.enemy.is_alive[0] = true;

  bool handler_called = false;
  event_manager_.Subscribe<PlayerEnemyCollisionEvent>(
      [&](const PlayerEnemyCollisionEvent&, EventContext&) {
        handler_called = true;
      });

  physics_manager_.StepPhysics(scene_, event_manager_);

  // Events are in the queue so calling Dispatch should fire the handler
  EventContext event_context{scene_};
  event_manager_.Dispatch(event_context);
  EXPECT_TRUE(handler_called);
}

TEST_F(PhysicsManagerTest, EventManager_FlushClearsStaleEventsBeforeNewStep) {
  // Simulate a stale event surviving from a "previous step"
  event_manager_.Emit(EnemyKilledEvent{99});
  ASSERT_EQ(event_manager_.GetEvents().size(), 1);

  // Flush (as StepGamePhysics does at start of each step)
  event_manager_.Flush();
  EXPECT_TRUE(event_manager_.GetEvents().empty());

  // New step produces its own events
  scene_.player.position_ = {100.0f, 100.0f};
  scene_.enemy.position[0] = {100.0f, 100.0f};
  scene_.enemy.is_alive[0] = true;
  scene_.enemy.attack_cooldown[0] = -1.0f;

  physics_manager_.StepPhysics(scene_, event_manager_);

  // Only events from this step should be present (no stale enemy_idx=99)
  for (const auto& e : event_manager_.GetEvents()) {
    if (std::holds_alternative<EnemyKilledEvent>(e)) {
      EXPECT_NE(std::get<EnemyKilledEvent>(e).enemy_idx, 99);
    }
  }
}

}  // namespace
}  // namespace arelto
