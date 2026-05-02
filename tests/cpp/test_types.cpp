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

}  // namespace
}  // namespace arelto
