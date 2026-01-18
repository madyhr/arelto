// tests/cpp/test_helpers.h
// Common test fixtures and utilities for rl2 unit tests

#ifndef RL2_TEST_HELPERS_H_
#define RL2_TEST_HELPERS_H_

#include <gtest/gtest.h>

#include "constants/enemy.h"
#include "constants/map.h"
#include "constants/player.h"
#include "entity.h"
#include "scene.h"
#include "types.h"

namespace rl2 {
namespace testing {

// Create a Scene with predictable initial state
inline Scene CreateTestScene() {
  Scene scene;
  scene.Reset();
  return scene;
}

// Create a player at a specific position
inline Player CreatePlayerAt(float x, float y) {
  Player player;
  player.position_ = {x, y};
  player.prev_position_ = player.position_;
  player.velocity_ = {0.0f, 0.0f};
  player.stats_.sprite_size = {kPlayerSpriteWidth, kPlayerSpriteHeight};
  player.stats_.health = kPlayerInitMaxHealth;
  player.stats_.max_health = kPlayerInitMaxHealth;
  player.stats_.movement_speed = kPlayerSpeed;
  player.stats_.inv_mass = kPlayerInvMass;
  player.stats_.level = 0;
  player.stats_.exp_points = 0;
  player.stats_.exp_points_required = kPlayerInitialExpRequirement;
  player.collider_ = {{kPlayerColliderOffsetX, kPlayerColliderOffsetY},
                      {kPlayerColliderWidth, kPlayerColliderHeight}};
  return player;
}

// Set enemy at specific position with default values
inline void SetEnemyAt(Enemy& enemy, int index, float x, float y,
                       bool is_alive = true) {
  enemy.position[index] = {x, y};
  enemy.prev_position[index] = enemy.position[index];
  enemy.velocity[index] = {0.0f, 0.0f};
  enemy.is_alive[index] = is_alive;
  enemy.is_done[index] = false;
  enemy.health_points[index] = kEnemyHealth;
  enemy.movement_speed[index] = kEnemySpeed;
  enemy.sprite_size[index] = {kEnemySpriteWidth, kEnemySpriteHeight};
  enemy.collider[index] = {{kEnemyColliderOffsetX, kEnemyColliderOffsetY},
                           {kEnemyColliderWidth, kEnemyColliderHeight}};
  enemy.inv_mass[index] = kEnemyInvMass;
  enemy.attack_cooldown[index] = 0.0f;
  enemy.attack_damage[index] = 1;
  enemy.damage_dealt_sim_step[index] = 0;
  enemy.timeout_timer[index] = 0.0f;
  enemy.is_truncated_latched[index] = false;
  enemy.is_terminated_latched[index] = false;
}

// Deactivate all enemies
inline void DeactivateAllEnemies(Enemy& enemy) {
  std::fill(enemy.is_alive.begin(), enemy.is_alive.end(), false);
}

// Initialize all enemies at default positions
inline void InitializeAllEnemies(Enemy& enemy) {
  for (int i = 0; i < kNumEnemies; ++i) {
    SetEnemyAt(enemy, i, kEnemyInitX + i * 50.0f, kEnemyInitY);
  }
}

// Create a projectile at specific position
inline ProjectileData CreateProjectileAt(float x, float y, float vx, float vy,
                                         float speed = 100.0f) {
  ProjectileData proj;
  proj.owner_id = 0;
  proj.position = {x, y};
  proj.velocity = {vx, vy};
  proj.speed = speed;
  proj.collider = {{8.0f, 8.0f}, {16, 16}};
  proj.sprite_size = {16, 16};
  proj.inv_mass = 1.0f;
  proj.proj_type = 0;
  return proj;
}

// Compare Vector2D with tolerance
inline void ExpectVector2DEq(const Vector2D& a, const Vector2D& b,
                             float tolerance = 1e-5f) {
  EXPECT_NEAR(a.x, b.x, tolerance);
  EXPECT_NEAR(a.y, b.y, tolerance);
}

// Check if Vector2D values are approximately equal
inline bool Vector2DApproxEq(const Vector2D& a, const Vector2D& b,
                             float tolerance = 1e-5f) {
  return std::abs(a.x - b.x) < tolerance && std::abs(a.y - b.y) < tolerance;
}

}  // namespace testing
}  // namespace rl2

#endif  // RL2_TEST_HELPERS_H_
