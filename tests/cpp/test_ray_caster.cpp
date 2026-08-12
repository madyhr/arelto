// Unit tests for ray-casting and ray-history behavior.

#include <gtest/gtest.h>

#include "constants/map.h"
#include "ray_caster.h"
#include "types.h"

namespace arelto {
namespace {

constexpr float kCellSize = static_cast<float>(kOccupancyMapResolution);
constexpr Vector2D kCellFourCenter = {4.5f * kCellSize, 4.5f * kCellSize};

class RayCasterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    occupancy_map_.Clear();
    occupancy_map_.AddBorder(EntityType::terrain);
  }

  void SetWall(int x, int y) { occupancy_map_.Set(x, y, EntityType::terrain); }

  DualRayHit Cast(Vector2D start, Vector2D direction) const {
    return CastRay({start, direction}, occupancy_map_);
  }

  FixedMap<kOccupancyMapWidth, kOccupancyMapHeight> occupancy_map_;
};

void FillHistory(EnemyRayCaster& ray_caster, float blocking_distance,
                 float non_blocking_distance) {
  for (auto& frame : ray_caster.ray_hit_distances) {
    for (auto& ray : frame) {
      ray.fill(blocking_distance);
    }
  }
  for (auto& frame : ray_caster.ray_hit_types) {
    for (auto& ray : frame) {
      ray.fill(EntityType::player);
    }
  }
  for (auto& frame : ray_caster.non_blocking_ray_hit_distances) {
    for (auto& ray : frame) {
      ray.fill(non_blocking_distance);
    }
  }
  for (auto& frame : ray_caster.non_blocking_ray_hit_types) {
    for (auto& ray : frame) {
      ray.fill(EntityType::projectile);
    }
  }
}

TEST_F(RayCasterTest, EmptyInteriorHitsMapBorder) {
  DualRayHit hit = Cast(kCellFourCenter, {1.0f, 0.0f});

  float expected_distance =
      static_cast<float>(kOccupancyMapWidth - 1) * kCellSize -
      kCellFourCenter.x;
  EXPECT_EQ(hit.blocking_hit.entity_type, EntityType::terrain);
  EXPECT_NEAR(hit.blocking_hit.distance, expected_distance, 1e-4f);
  EXPECT_EQ(hit.non_blocking_hit.entity_type, EntityType::None);
}

TEST_F(RayCasterTest, HorizontalRayDetectsWall) {
  SetWall(6, 4);

  DualRayHit hit = Cast(kCellFourCenter, {1.0f, 0.0f});

  EXPECT_EQ(hit.blocking_hit.entity_type, EntityType::terrain);
  EXPECT_NEAR(hit.blocking_hit.distance, 1.5f * kCellSize, 1e-4f);
}

TEST_F(RayCasterTest, VerticalRayDetectsWall) {
  SetWall(4, 6);

  DualRayHit hit = Cast(kCellFourCenter, {0.0f, 1.0f});

  EXPECT_EQ(hit.blocking_hit.entity_type, EntityType::terrain);
  EXPECT_NEAR(hit.blocking_hit.distance, 1.5f * kCellSize, 1e-4f);
}

TEST_F(RayCasterTest, NegativeDirectionDetectsWall) {
  SetWall(2, 4);

  DualRayHit hit = Cast(kCellFourCenter, {-1.0f, 0.0f});

  EXPECT_EQ(hit.blocking_hit.entity_type, EntityType::terrain);
  EXPECT_NEAR(hit.blocking_hit.distance, 1.5f * kCellSize, 1e-4f);
}

TEST_F(RayCasterTest, DiagonalRayDetectsWall) {
  SetWall(6, 6);
  SetWall(6, 5);
  SetWall(5, 6);

  DualRayHit hit = Cast(kCellFourCenter, Vector2D{1.0f, 1.0f}.Normalized());

  EXPECT_EQ(hit.blocking_hit.entity_type, EntityType::terrain);
  EXPECT_GT(hit.blocking_hit.distance, 0.0f);
}

TEST_F(RayCasterTest, DetectsWallAtCloseProximity) {
  SetWall(5, 4);

  DualRayHit hit =
      Cast({5.0f * kCellSize - 1.0f, kCellFourCenter.y}, {1.0f, 0.0f});

  EXPECT_EQ(hit.blocking_hit.entity_type, EntityType::terrain);
  EXPECT_NEAR(hit.blocking_hit.distance, 1.0f, 1e-4f);
}

TEST_F(RayCasterTest, DetectsProjectileBeforeBlockingWall) {
  occupancy_map_.Add(5, 4, EntityType::projectile);
  SetWall(6, 4);

  DualRayHit hit = Cast(kCellFourCenter, {1.0f, 0.0f});

  EXPECT_EQ(hit.non_blocking_hit.entity_type, EntityType::projectile);
  EXPECT_NEAR(hit.non_blocking_hit.distance, 0.5f * kCellSize, 1e-4f);
  EXPECT_EQ(hit.blocking_hit.entity_type, EntityType::terrain);
  EXPECT_NEAR(hit.blocking_hit.distance, 1.5f * kCellSize, 1e-4f);
}

TEST(RayHistoryTest, FindsEntityTypeInSelectedFrameAndEnemy) {
  RayHistoryTypes history{};
  history[0][2][3] = EntityType::terrain;

  EXPECT_TRUE(IsEntityTypePresent(history, {0, 3}, EntityType::terrain));
  EXPECT_FALSE(IsEntityTypePresent(history, {0, 4}, EntityType::terrain));
  EXPECT_FALSE(IsEntityTypePresent(history, {0, 3}, EntityType::player));
}

TEST(RayHistoryTest, ResetClearsAllHistoryAndRewindsHead) {
  EnemyRayCaster ray_caster;
  FillHistory(ray_caster, 10.0f, 20.0f);
  ray_caster.history_idx = kRayHistoryLength - 1;

  ray_caster.Reset();

  EXPECT_EQ(ray_caster.ray_hit_distances, RayHistoryDistances{});
  EXPECT_EQ(ray_caster.ray_hit_types, RayHistoryTypes{});
  EXPECT_EQ(ray_caster.non_blocking_ray_hit_distances, RayHistoryDistances{});
  EXPECT_EQ(ray_caster.non_blocking_ray_hit_types, RayHistoryTypes{});
  EXPECT_EQ(ray_caster.history_idx, 0);
}

TEST(RayHistoryTest, ResetEnemyClearsOnlySelectedEnemy) {
  constexpr int kResetEnemy = 0;
  constexpr int kUnaffectedEnemy = kNumEnemies - 1;
  constexpr float kBlockingDistance = 10.0f;
  constexpr float kNonBlockingDistance = 20.0f;

  EnemyRayCaster ray_caster;
  FillHistory(ray_caster, kBlockingDistance, kNonBlockingDistance);
  ray_caster.history_idx = kRayHistoryLength - 1;

  ray_caster.ResetEnemy(kResetEnemy);

  for (int history = 0; history < kRayHistoryLength; ++history) {
    for (int ray = 0; ray < kNumRays; ++ray) {
      SCOPED_TRACE(::testing::Message()
                   << "history=" << history << ", ray=" << ray);

      EXPECT_FLOAT_EQ(ray_caster.ray_hit_distances[history][ray][kResetEnemy],
                      0.0f);
      EXPECT_EQ(ray_caster.ray_hit_types[history][ray][kResetEnemy],
                EntityType::None);
      EXPECT_FLOAT_EQ(
          ray_caster.non_blocking_ray_hit_distances[history][ray][kResetEnemy],
          0.0f);
      EXPECT_EQ(
          ray_caster.non_blocking_ray_hit_types[history][ray][kResetEnemy],
          EntityType::None);

      EXPECT_FLOAT_EQ(
          ray_caster.ray_hit_distances[history][ray][kUnaffectedEnemy],
          kBlockingDistance);
      EXPECT_EQ(ray_caster.ray_hit_types[history][ray][kUnaffectedEnemy],
                EntityType::player);
      EXPECT_FLOAT_EQ(
          ray_caster
              .non_blocking_ray_hit_distances[history][ray][kUnaffectedEnemy],
          kNonBlockingDistance);
      EXPECT_EQ(
          ray_caster.non_blocking_ray_hit_types[history][ray][kUnaffectedEnemy],
          EntityType::projectile);
    }
  }
  EXPECT_EQ(ray_caster.history_idx, kRayHistoryLength - 1);
}

}  // namespace
}  // namespace arelto
