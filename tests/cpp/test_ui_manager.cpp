// tests/cpp/test_ui_manager.cpp
// Unit tests for UIManager class

#include <gtest/gtest.h>

#include "scene.h"
#include "test_helpers.h"
#include "ui_manager.h"

namespace arelto {
namespace {

class UIManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = testing::CreateTestScene();
    ui_manager_.SetupUI();
  }

  Scene scene_;
  UIManager ui_manager_;
};

// =============================================================================
// UpdateHealthBar Tests
// =============================================================================

TEST_F(UIManagerTest, UpdateHealthBar_FillWidthChangesWithHealth) {
  UIElement* fill = ui_manager_.health_bar_.GetElemByTag(UIElement::Tag::fill);
  ASSERT_NE(fill, nullptr);

  // Set full health
  scene_.player.stats_.health = scene_.player.stats_.max_health;
  ui_manager_.UpdateHealthBar(scene_);
  uint32_t full_health_width = fill->sprite_size.width;

  // Set half health
  scene_.player.stats_.health = scene_.player.stats_.max_health / 2;
  ui_manager_.UpdateHealthBar(scene_);
  uint32_t half_health_width = fill->sprite_size.width;

  // Full health should show wider bar than half health
  EXPECT_EQ(full_health_width / 2, half_health_width);
}

TEST_F(UIManagerTest, UpdateHealthBar_HandlesZeroHealth) {
  scene_.player.stats_.health = 0;

  EXPECT_NO_THROW(ui_manager_.UpdateHealthBar(scene_));

  UIElement* fill = ui_manager_.health_bar_.GetElemByTag(UIElement::Tag::fill);
  ASSERT_NE(fill, nullptr);
  // Width should be 0
  EXPECT_EQ(fill->sprite_size.width, 0u);
}

// =============================================================================
// UpdateExpBar Tests
// =============================================================================

TEST_F(UIManagerTest, UpdateExpBar_FillWidthChangesWithExp) {
  UIElement* fill = ui_manager_.exp_bar_.GetElemByTag(UIElement::Tag::fill);
  ASSERT_NE(fill, nullptr);

  // Set no exp
  scene_.player.stats_.exp_points = 0;
  scene_.player.stats_.exp_points_required = 100;
  ui_manager_.UpdateExpBar(scene_);
  uint32_t zero_exp_width = fill->sprite_size.width;

  // Set half exp
  scene_.player.stats_.exp_points = 50;
  ui_manager_.UpdateExpBar(scene_);
  uint32_t half_exp_width = fill->sprite_size.width;

  // Half exp should show wider bar than zero exp
  EXPECT_GT(half_exp_width, zero_exp_width);
}

// =============================================================================
// UpdateTimer Tests
// =============================================================================

TEST_F(UIManagerTest, UpdateTimer_StoresRawSeconds) {
  ui_manager_.UpdateTimer(65.0f);  // 65 seconds

  UIElement* text = ui_manager_.timer_.GetElemByTag(UIElement::Tag::text);
  ASSERT_NE(text, nullptr);

  // Timer stores raw seconds as string
  EXPECT_EQ(text->text_value, "65");
}

TEST_F(UIManagerTest, UpdateTimer_HandlesZeroTime) {
  ui_manager_.UpdateTimer(0.0f);

  UIElement* text = ui_manager_.timer_.GetElemByTag(UIElement::Tag::text);
  ASSERT_NE(text, nullptr);

  EXPECT_EQ(text->text_value, "0");
}

// =============================================================================
// UpdateLevelIndicator Tests
// =============================================================================

TEST_F(UIManagerTest, UpdateLevelIndicator_UpdatesText) {
  scene_.player.stats_.level = 5;
  ui_manager_.UpdateLevelIndicator(scene_);

  UIElement* text =
      ui_manager_.level_indicator_.GetElemByTag(UIElement::Tag::text);
  ASSERT_NE(text, nullptr);

  // Text should contain the level number
  EXPECT_NE(text->text_value.find("5"), std::string::npos);
}

}  // namespace
}  // namespace arelto
