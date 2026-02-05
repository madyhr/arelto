// tests/cpp/test_progression_manager.cpp
// Unit tests for ProgressionManager class

#include <gtest/gtest.h>

#include <memory>

#include "constants/progression_manager.h"
#include "progression_manager.h"
#include "scene.h"
#include "test_helpers.h"
#include "upgrades.h"

namespace arelto {
namespace {

class ProgressionManagerTest : public ::testing::Test {
 protected:
  void SetUp() override { scene_ = testing::CreateTestScene(); }

  Scene scene_;
  ProgressionManager progression_manager_;
};

// =============================================================================
// CheckLevelUp Tests
// =============================================================================

TEST_F(ProgressionManagerTest, CheckLevelUp_WorksCorrectly) {
  scene_.player.stats_.exp_points_required = 100;

  // True cases
  scene_.player.stats_.exp_points = 100;
  EXPECT_TRUE(progression_manager_.CheckLevelUp(scene_.player));

  scene_.player.stats_.exp_points = 150;
  EXPECT_TRUE(progression_manager_.CheckLevelUp(scene_.player));

  // False cases
  scene_.player.stats_.exp_points = 99;
  EXPECT_FALSE(progression_manager_.CheckLevelUp(scene_.player));

  scene_.player.stats_.exp_points = 0;
  EXPECT_FALSE(progression_manager_.CheckLevelUp(scene_.player));

  scene_.player.stats_.exp_points = -100;
  EXPECT_FALSE(progression_manager_.CheckLevelUp(scene_.player));
}

// =============================================================================
// GenerateLevelUpOptions Tests
// =============================================================================

TEST_F(ProgressionManagerTest, GenerateLevelUpOptions_CreatesCorrectCount) {
  progression_manager_.GenerateLevelUpOptions(scene_);

  EXPECT_EQ(scene_.level_up_options.size(),
            static_cast<size_t>(kNumUpgradeOptions));
}

TEST_F(ProgressionManagerTest, GenerateLevelUpOptions_ClearsExistingOptions) {
  // Add some dummy options
  scene_.level_up_options.push_back(nullptr);
  scene_.level_up_options.push_back(nullptr);
  ASSERT_EQ(scene_.level_up_options.size(), 2);

  progression_manager_.GenerateLevelUpOptions(scene_);

  // Should have exactly kNumUpgradeOptions, not more
  EXPECT_EQ(scene_.level_up_options.size(),
            static_cast<size_t>(kNumUpgradeOptions));
}

TEST_F(ProgressionManagerTest, GenerateLevelUpOptions_CreatesValidOptions) {
  progression_manager_.GenerateLevelUpOptions(scene_);

  for (const auto& option : scene_.level_up_options) {
    ASSERT_NE(option, nullptr);
    EXPECT_FALSE(option->GetDescription().empty());
    EXPECT_FALSE(option->GetSpellName().empty());

    UpgradeType type = option->GetType();
    EXPECT_LT(static_cast<int>(type), static_cast<int>(UpgradeType::count));
  }
}

// =============================================================================
// ApplyUpgrade Tests
// =============================================================================

TEST_F(ProgressionManagerTest, ApplyUpgrade_IncreasesLevel) {
  scene_.player.stats_.level = 0;
  scene_.player.stats_.exp_points = 100;
  scene_.player.stats_.exp_points_required = 100;

  progression_manager_.GenerateLevelUpOptions(scene_);
  progression_manager_.ApplyUpgrade(scene_, 0);

  EXPECT_EQ(scene_.player.stats_.level, 1);
}

TEST_F(ProgressionManagerTest, ApplyUpgrade_DeductsExp) {
  scene_.player.stats_.exp_points = 150;
  scene_.player.stats_.exp_points_required = 100;

  progression_manager_.GenerateLevelUpOptions(scene_);
  progression_manager_.ApplyUpgrade(scene_, 0);

  // Exp should be reduced by the required amount
  EXPECT_EQ(scene_.player.stats_.exp_points, 50);
}

TEST_F(ProgressionManagerTest, ApplyUpgrade_ScalesExpRequired) {
  scene_.player.stats_.exp_points = 100;
  scene_.player.stats_.exp_points_required = 100;

  int initial_exp_required = scene_.player.stats_.exp_points_required;

  progression_manager_.GenerateLevelUpOptions(scene_);
  progression_manager_.ApplyUpgrade(scene_, 0);

  int new_exp_required =
      progression_manager_.ApplyExpScalingLaw(initial_exp_required);

  EXPECT_EQ(scene_.player.stats_.exp_points_required, new_exp_required);
}

TEST_F(ProgressionManagerTest, ApplyUpgrade_InvalidIndex_Negative_NoOp) {
  scene_.player.stats_.level = 0;

  progression_manager_.GenerateLevelUpOptions(scene_);
  progression_manager_.ApplyUpgrade(scene_, -1);

  // Level should not have changed
  EXPECT_EQ(scene_.player.stats_.level, 0);
}

TEST_F(ProgressionManagerTest, ApplyUpgrade_WorksForAllValidIndices) {
  for (int i = 0; i < kNumUpgradeOptions; ++i) {
    scene_ = testing::CreateTestScene();
    scene_.player.stats_.level = 0;
    scene_.player.stats_.exp_points = 1000;
    scene_.player.stats_.exp_points_required = 100;

    progression_manager_.GenerateLevelUpOptions(scene_);
    progression_manager_.ApplyUpgrade(scene_, i);

    EXPECT_EQ(scene_.player.stats_.level, 1)
        << "ApplyUpgrade failed for index " << i;
  }
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(ProgressionManagerTest, ApplyUpgrade_ChangesPlayerStats) {
  // Setup player with known initial stats
  scene_.player.stats_.level = 0;
  scene_.player.stats_.exp_points = 1000;
  scene_.player.stats_.exp_points_required = 100;
  // Ensure spell stats are initialized (cooldowns, damages, etc.)
  scene_.player.UpdateAllSpellStats();

  // Manually create a deterministic upgrade option
  // We avoid using GenerateLevelUpOptions to remove RNG.
  SpellId target_spell = SpellId::FireballId;
  std::string spell_name = "Fireball";
  float initial_damage =
      static_cast<float>(scene_.player.spell_stats_.damage[target_spell]);
  float new_damage = initial_damage + 10.0f;

  auto upgrade = std::make_unique<SpellStatUpgrade>(target_spell, spell_name,
                                                    UpgradeType::damage,
                                                    initial_damage, new_damage);

  scene_.level_up_options.clear();
  scene_.level_up_options.push_back(std::move(upgrade));

  // Apply the upgrade (index 0)
  progression_manager_.ApplyUpgrade(scene_, 0);

  // Verify
  float actual_damage =
      static_cast<float>(scene_.player.spell_stats_.damage[target_spell]);
  EXPECT_GT(actual_damage, initial_damage);
  EXPECT_FLOAT_EQ(actual_damage, new_damage);
}

}  // namespace
}  // namespace arelto
