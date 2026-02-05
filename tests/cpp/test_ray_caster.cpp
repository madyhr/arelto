// tests/cpp/test_ray_caster.cpp
// Unit tests for RayCaster functions

#include <gtest/gtest.h>
#include <cmath>

#include "constants/map.h"
#include "ray_caster.h"
#include "types.h"

namespace arelto {
namespace {

class RayCasterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize an empty map (all none)
    // We can use a lambda or loop to set specific tiles for tests
    occupancy_map_.Clear();
    AddBorders();
  }

  void AddBorders() {
    for (int x = 0; x < kOccupancyMapWidth; ++x) {
      SetWall(x, 0);
      SetWall(x, kOccupancyMapHeight - 1);
    }
    for (int y = 0; y < kOccupancyMapHeight; ++y) {
      SetWall(0, y);
      SetWall(kOccupancyMapWidth - 1, y);
    }
  }

  void ClearMap() { occupancy_map_.Clear(); }

  void SetWall(int x, int y) { occupancy_map_.Set(x, y, EntityType::terrain); }

  FixedMap<kOccupancyMapWidth, kOccupancyMapHeight> occupancy_map_;
};

// =============================================================================
// Helper Functions
// =============================================================================

// Helper to check if a float is approximately equal
bool FloatEq(float a, float b, float epsilon = 1e-4f) {
  return std::abs(a - b) < epsilon;
}

// =============================================================================
// CastRay Tests
// =============================================================================

TEST_F(RayCasterTest, CastRay_NoObstacles_ReturnsLargeDistance) {
  // Start in the middle of a large empty area
  Vector2D start_pos = {100.0f, 100.0f};
  Vector2D ray_dir = {1.0f, 0.0f};  // Right

  // Since map is bounded, it should eventually hit the "virtual" world bounds
  // or loop forever if the caster doesn't handle OOB.
  // However, CastRay doc says: "This function assumes that the occupancy map is surrounded by grid cells that have an EntityType other than None."
  // So we MUST set bounds to walls to prevent infinite loops if the implementation expects it.

  // Let's protect our test by setting boundaries.
  // No need to set manual borders here as SetUp does it.

  RayHit hit = CastRay(start_pos, ray_dir, occupancy_map_);

  // Should hit the right wall (from SetUp borders)
  // Map width is kOccupancyMapWidth * Resolution.
  // Bounds are at x=0 and x=Width-1.
  // Ray goes Right -> Hits x=Width-1.
  // Start x=100 (Grid 4). Wall at Width-1 (Grid 76).
  // Dist approx (76 - 4) * 25 = 72 * 25 = 1800.

  EXPECT_EQ(hit.entity_type, EntityType::terrain);
  EXPECT_GT(hit.distance, 1000.0f);
}

TEST_F(RayCasterTest, CastRay_OrthogonalX_DetectsWall) {
  Vector2D start_pos = {100.0f, 100.0f};  // Grid (4, 4) with Res=25

  // Place wall at Grid (6, 4)
  SetWall(6, 4);

  // Expected distance:
  // Start x=100. Grid 4.
  // Wall at Grid 6. Wall starts at x = 6 * 25 = 150.
  // Distance = 150 - 100 = 50.

  Vector2D ray_dir = {1.0f, 0.0f};  // Right

  RayHit hit = CastRay(start_pos, ray_dir, occupancy_map_);

  EXPECT_EQ(hit.entity_type, EntityType::terrain);
  EXPECT_NEAR(hit.distance, 50.0f, 1.0f);
}

TEST_F(RayCasterTest, CastRay_OrthogonalY_DetectsWall) {
  Vector2D start_pos = {100.0f, 100.0f};  // Grid (4, 4)
  // Place wall at Grid (4, 6)
  SetWall(4, 6);

  // Wall starts at y = 6 * 25 = 150.
  // Dist = 150 - 100 = 50.

  Vector2D ray_dir = {0.0f, 1.0f};  // Down

  RayHit hit = CastRay(start_pos, ray_dir, occupancy_map_);

  EXPECT_EQ(hit.entity_type, EntityType::terrain);
  EXPECT_NEAR(hit.distance, 50.0f, 1.0f);
}

TEST_F(RayCasterTest, CastRay_NegativeDirection_DetectsWall) {
  Vector2D start_pos = {100.0f, 100.0f};  // Grid (4, 4)
  // Place wall at Grid (2, 4)
  SetWall(2, 4);

  // Wall checks:
  // Ray Left.
  // Grid 4 -> 3 -> 2.
  // Wall at 2. Wall (right edge) is at (2+1)*25 = 75?
  // Or simply: Ray hits the side x=3 boundary of cell 2?
  // Let's trace:
  // Start 100. Dir -1.
  // Next X line: 4 * 25 = 100. Distance 0?
  // No, if start is exactly 100, might be tricky.
  // Let's move start slightly to 110. Grid 4 (range 100-125).
  start_pos = {112.5f, 112.5f};  // Center of cell (4,4)

  // Wall at (2, 4).
  // Wall right edge x = (2+1)*25 = 75.
  // Dist = 112.5 - 75 = 37.5.

  Vector2D ray_dir = {-1.0f, 0.0f};  // Left

  RayHit hit = CastRay(start_pos, ray_dir, occupancy_map_);

  EXPECT_EQ(hit.entity_type, EntityType::terrain);
  EXPECT_NEAR(hit.distance, 37.5f, 1.0f);
}

TEST_F(RayCasterTest, CastRay_Diagonal_DetectsWall) {
  // Start at (4.5, 4.5) grid units -> 112.5
  Vector2D start_pos = {112.5f, 112.5f};

  // Wall at (6, 6).
  SetWall(6, 6);
  // Fill Corner gap
  SetWall(6, 5);
  SetWall(5, 6);

  // Direction diagonal (1, 1).
  Vector2D ray_dir = {1.0f, 1.0f};
  ray_dir = ray_dir.Normalized();

  RayHit hit = CastRay(start_pos, ray_dir, occupancy_map_);

  EXPECT_EQ(hit.entity_type, EntityType::terrain);
  EXPECT_GT(hit.distance, 0.0f);
}

TEST_F(RayCasterTest, CastRay_CloseProximity_DetectsImmediateWall) {
  // Wall at (5, 4)
  SetWall(5, 4);

  // Player at x=124.0 (Grid 4.96). Near right edge of 4.
  // y=112.5 (Center of 4).
  Vector2D start_pos = {124.0f, 112.5f};
  Vector2D ray_dir = {1.0f, 0.0f};

  RayHit hit = CastRay(start_pos, ray_dir, occupancy_map_);

  EXPECT_EQ(hit.entity_type, EntityType::terrain);
  // Wall starts at 5*25=125. Start=124. Dist=1.0.
  EXPECT_NEAR(hit.distance, 1.0f, 0.1f);
}

TEST_F(RayCasterTest, IsEntityTypePresent_Found) {
  RayHistoryTypes history;
  // Initialize with None
  for (auto& frame : history) {
    for (auto& ray : frame) {
      ray.fill(EntityType::None);
    }
  }

  // Set one to wall
  history[0][0][0] = EntityType::terrain;

  EXPECT_TRUE(IsEntityTypePresent(history, 0, 0, EntityType::terrain));
}

TEST_F(RayCasterTest, IsEntityTypePresent_NotFound) {
  RayHistoryTypes history;
  for (auto& frame : history) {
    for (auto& ray : frame) {
      ray.fill(EntityType::None);
    }
  }

  EXPECT_FALSE(IsEntityTypePresent(history, 0, 0, EntityType::terrain));
}

}  // namespace
}  // namespace arelto
