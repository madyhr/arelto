// tests/cpp/test_helpers.h
// Common test fixtures and utilities for rl2 unit tests

#ifndef RL2_TEST_HELPERS_H_
#define RL2_TEST_HELPERS_H_

#include <gtest/gtest.h>

#include "entity.h"
#include "event_manager.h"
#include "scene.h"
#include "spell_manager.h"
#include "types.h"

namespace arelto {
namespace testing {

// Create a Scene with predictable initial state
inline Scene CreateTestScene() {
  static SpellManager spell_manager;
  if (!spell_manager.GetSpellCount()) {
    spell_manager.Initialize();
  }
  Scene scene;
  scene.player.SetSpellManager(&spell_manager);
  scene.player.spell_stats_.Resize(spell_manager.GetSpellCount());
  scene.Reset(MakeDefaultEntityConfig());
  return scene;
}

// Build an EventContext for tests. Centralized so future EventContext field
// additions only require updating this helper.
inline EventContext MakeEventContext(Scene& scene,
                                     EventManager& event_manager) {
  return EventContext{event_manager, scene};
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
}  // namespace arelto

#endif  // RL2_TEST_HELPERS_H_
