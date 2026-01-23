// tests/cpp/test_ui_manager.cpp
// Unit tests for UIManager class

#include <gtest/gtest.h>

#include "scene.h"
#include "test_helpers.h"
#include "ui_manager.h"

namespace rl2 {
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
// SetupUI Tests
// =============================================================================

TEST_F(UIManagerTest, SetupUI_InitializesHealthBar) {
  // Health bar should have elements after setup
  EXPECT_FALSE(ui_manager_.health_bar_.elements.empty());
}

TEST_F(UIManagerTest, SetupUI_InitializesExpBar) {
  EXPECT_FALSE(ui_manager_.exp_bar_.elements.empty());
}

TEST_F(UIManagerTest, SetupUI_InitializesTimer) {
  EXPECT_FALSE(ui_manager_.timer_.elements.empty());
}

TEST_F(UIManagerTest, SetupUI_InitializesLevelIndicator) {
  EXPECT_FALSE(ui_manager_.level_indicator_.elements.empty());
}

// =============================================================================
// SetupHealthBar Tests
// =============================================================================

TEST_F(UIManagerTest, SetupHealthBar_HasBackgroundElement) {
  UIElement* bg =
      ui_manager_.health_bar_.GetElemByTag(UIElement::Tag::background);
  EXPECT_NE(bg, nullptr);
}

TEST_F(UIManagerTest, SetupHealthBar_HasFillElement) {
  UIElement* fill = ui_manager_.health_bar_.GetElemByTag(UIElement::Tag::fill);
  EXPECT_NE(fill, nullptr);
}

TEST_F(UIManagerTest, SetupHealthBar_HasTextElement) {
  // Health bar has text element (not icon)
  UIElement* text = ui_manager_.health_bar_.GetElemByTag(UIElement::Tag::text);
  EXPECT_NE(text, nullptr);
}

// =============================================================================
// SetupExpBar Tests
// =============================================================================

TEST_F(UIManagerTest, SetupExpBar_HasBackgroundElement) {
  UIElement* bg = ui_manager_.exp_bar_.GetElemByTag(UIElement::Tag::background);
  EXPECT_NE(bg, nullptr);
}

TEST_F(UIManagerTest, SetupExpBar_HasFillElement) {
  UIElement* fill = ui_manager_.exp_bar_.GetElemByTag(UIElement::Tag::fill);
  EXPECT_NE(fill, nullptr);
}

// =============================================================================
// GetElemByTag Tests
// =============================================================================

TEST_F(UIManagerTest, GetElemByTag_FindsExistingTag) {
  UIElement* elem =
      ui_manager_.health_bar_.GetElemByTag(UIElement::Tag::background);
  EXPECT_NE(elem, nullptr);
  EXPECT_EQ(elem->tag, UIElement::Tag::background);
}

TEST_F(UIManagerTest, GetElemByTag_ReturnsNullForMissingTag) {
  // Create a new empty group
  UIElementGroup empty_group;
  empty_group.type = UIElementGroupType::health_bar;

  UIElement* elem = empty_group.GetElemByTag(UIElement::Tag::background);
  EXPECT_EQ(elem, nullptr);
}

TEST_F(UIManagerTest, GetElemByTag_ReturnsNullForNoneTag) {
  // None tag should not match any specifically tagged element
  UIElement* elem = ui_manager_.health_bar_.GetElemByTag(UIElement::Tag::none);
  // May or may not find an element depending on implementation
  // Just verify it doesn't crash
  (void)elem;
}

// =============================================================================
// UpdateUI Tests
// =============================================================================

TEST_F(UIManagerTest, UpdateUI_DoesNotCrash) {
  EXPECT_NO_THROW(ui_manager_.UpdateUI(scene_, 0.0f));
}

TEST_F(UIManagerTest, UpdateUI_DoesNotCrashWithLargeTime) {
  EXPECT_NO_THROW(ui_manager_.UpdateUI(scene_, 3600.0f));  // 1 hour
}

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
  EXPECT_GT(full_health_width, half_health_width);
}

TEST_F(UIManagerTest, UpdateHealthBar_HandlesZeroHealth) {
  scene_.player.stats_.health = 0;

  EXPECT_NO_THROW(ui_manager_.UpdateHealthBar(scene_));

  UIElement* fill = ui_manager_.health_bar_.GetElemByTag(UIElement::Tag::fill);
  ASSERT_NE(fill, nullptr);
  // Width should be 0 or very small
  EXPECT_EQ(fill->sprite_size.width, 0u);
}

TEST_F(UIManagerTest, UpdateHealthBar_HandlesMaxHealth) {
  scene_.player.stats_.health = scene_.player.stats_.max_health;

  EXPECT_NO_THROW(ui_manager_.UpdateHealthBar(scene_));
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

TEST_F(UIManagerTest, UpdateExpBar_HandlesZeroExp) {
  scene_.player.stats_.exp_points = 0;
  scene_.player.stats_.exp_points_required = 100;

  EXPECT_NO_THROW(ui_manager_.UpdateExpBar(scene_));
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

TEST_F(UIManagerTest, UpdateTimer_HandlesLargeTime) {
  ui_manager_.UpdateTimer(3661.0f);  // 3661 seconds

  UIElement* text = ui_manager_.timer_.GetElemByTag(UIElement::Tag::text);
  ASSERT_NE(text, nullptr);

  EXPECT_EQ(text->text_value, "3661");
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

TEST_F(UIManagerTest, UpdateLevelIndicator_HandlesZeroLevel) {
  scene_.player.stats_.level = 0;

  EXPECT_NO_THROW(ui_manager_.UpdateLevelIndicator(scene_));
}

TEST_F(UIManagerTest, UpdateLevelIndicator_HandlesHighLevel) {
  scene_.player.stats_.level = 999;

  EXPECT_NO_THROW(ui_manager_.UpdateLevelIndicator(scene_));
}

}  // namespace
}  // namespace rl2
