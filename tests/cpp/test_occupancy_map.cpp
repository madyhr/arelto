// tests/cpp/test_occupancy_map.cpp
#include <gtest/gtest.h>
#include "map.h"
#include "types.h"

namespace arelto {
namespace {

class OccupancyMapTest : public ::testing::Test {
 protected:
  FixedMap<10, 10> map_;
};

TEST_F(OccupancyMapTest, InitialStateIsNone) {
  for (int x = 0; x < 10; ++x) {
    for (int y = 0; y < 10; ++y) {
      EXPECT_EQ(map_.Get(x, y), EntityType::None);
      EXPECT_EQ(map_.GetMask(x, y), kMaskTypeNone);
    }
  }
}

TEST_F(OccupancyMapTest, AddSetsBitsCorrectly) {
  map_.Add(0, 0, EntityType::player);
  EXPECT_EQ(map_.GetMask(0, 0), kMaskTypePlayer);
  EXPECT_EQ(map_.Get(0, 0), EntityType::player);

  map_.Add(0, 0, EntityType::projectile);
  EXPECT_EQ(map_.GetMask(0, 0), kMaskTypePlayer | kMaskTypeProjectile);

  // Get() should prioritize Player over Projectile
  EXPECT_EQ(map_.Get(0, 0), EntityType::player);
}

TEST_F(OccupancyMapTest, RemoveClearsBitsCorrectly) {
  map_.Add(0, 0, EntityType::player);
  map_.Add(0, 0, EntityType::projectile);

  // Remove Player
  map_.Remove(0, 0, EntityType::player);
  EXPECT_EQ(map_.GetMask(0, 0), kMaskTypeProjectile);
  // Now Get() should return Projectile
  EXPECT_EQ(map_.Get(0, 0), EntityType::projectile);

  // Remove Projectile
  map_.Remove(0, 0, EntityType::projectile);
  EXPECT_EQ(map_.GetMask(0, 0), kMaskTypeNone);
  EXPECT_EQ(map_.Get(0, 0), EntityType::None);
}

TEST_F(OccupancyMapTest, SetBehavesLikeClearAndAdd) {
  map_.Add(0, 0, EntityType::player);

  // Set should overwrite
  map_.Set(0, 0, EntityType::enemy);

  EXPECT_EQ(map_.GetMask(0, 0), kMaskTypeEnemy);
  EXPECT_EQ(map_.Get(0, 0), EntityType::enemy);
}

TEST_F(OccupancyMapTest, PriorityLogicCorrectness) {
  // Add everything
  map_.Add(0, 0, EntityType::exp_gem);
  EXPECT_EQ(map_.Get(0, 0), EntityType::exp_gem);

  map_.Add(0, 0, EntityType::projectile);
  EXPECT_EQ(map_.Get(0, 0), EntityType::projectile);  // Projectile > Gem

  map_.Add(0, 0, EntityType::enemy);
  EXPECT_EQ(map_.Get(0, 0), EntityType::enemy);  // Enemy > Projectile

  map_.Add(0, 0, EntityType::terrain);
  EXPECT_EQ(map_.Get(0, 0), EntityType::terrain);  // Terrain > Enemy

  map_.Add(0, 0, EntityType::player);
  EXPECT_EQ(map_.Get(0, 0), EntityType::player);  // Player > Terrain
}

TEST_F(OccupancyMapTest, AddBorderWorksCorrectly) {
  map_.AddBorder(EntityType::terrain);

  // Corners
  EXPECT_EQ(map_.Get(0, 0), EntityType::terrain);
  EXPECT_EQ(map_.Get(9, 9), EntityType::terrain);

  // Middle should be empty
  EXPECT_EQ(map_.Get(5, 5), EntityType::None);
}

}  // namespace
}  // namespace arelto
