// Unit tests for shared gameplay types and helpers

#include <gtest/gtest.h>

#include "types.h"

namespace arelto {
namespace {

TEST(TypesTest, CreateCenteredCollider_AppliesDefaultMargin) {
  Collider collider = CreateCenteredCollider({60, 75});

  EXPECT_FLOAT_EQ(collider.offset.x, 30.0f);
  EXPECT_FLOAT_EQ(collider.offset.y, 37.5f);
  EXPECT_EQ(collider.size.width, 30);
  EXPECT_EQ(collider.size.height, 45);
}

TEST(TypesTest, CreateCenteredCollider_DoesNotUnderflowSmallSprites) {
  Collider collider = CreateCenteredCollider({25, 33});

  EXPECT_FLOAT_EQ(collider.offset.x, 12.5f);
  EXPECT_FLOAT_EQ(collider.offset.y, 16.5f);
  EXPECT_EQ(collider.size.width, 1);
  EXPECT_EQ(collider.size.height, 3);
}

TEST(TypesTest, CreateCenteredCollider_ZeroMarginUsesFullSpriteSize) {
  Collider collider = CreateCenteredCollider({25, 33}, 0);

  EXPECT_FLOAT_EQ(collider.offset.x, 12.5f);
  EXPECT_FLOAT_EQ(collider.offset.y, 16.5f);
  EXPECT_EQ(collider.size.width, 25);
  EXPECT_EQ(collider.size.height, 33);
}

TEST(TypesTest, StatsSize_ColliderTracksWidthModifiers) {
  StatsSize size(1.25f);
  size.SetBaseWidth(60.0f);

  Collider before = size.GetCollider();

  size.width_.AddModifier(Modifier{20.0f, ModifierType::flat, nullptr});

  Collider after = size.GetCollider();

  EXPECT_EQ(size.GetWidth(), 80);
  EXPECT_EQ(size.GetHeight(), 100);

  EXPECT_FLOAT_EQ(before.offset.x, 30.0f);
  EXPECT_FLOAT_EQ(before.offset.y, 37.5f);
  EXPECT_EQ(before.size.width, 30);
  EXPECT_EQ(before.size.height, 45);

  EXPECT_FLOAT_EQ(after.offset.x, 40.0f);
  EXPECT_FLOAT_EQ(after.offset.y, 50.0f);
  EXPECT_EQ(after.size.width, 50);
  EXPECT_EQ(after.size.height, 70);
}

}  // namespace
}  // namespace arelto
