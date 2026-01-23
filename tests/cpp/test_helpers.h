// tests/cpp/test_helpers.h
// Common test fixtures and utilities for rl2 unit tests

#ifndef RL2_TEST_HELPERS_H_
#define RL2_TEST_HELPERS_H_

#include <gtest/gtest.h>

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

// Deactivate all enemies
inline void DeactivateAllEnemies(Enemy& enemy) {
  std::fill(enemy.is_alive.begin(), enemy.is_alive.end(), false);
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

}  // namespace testing
}  // namespace rl2

#endif  // RL2_TEST_HELPERS_H_
