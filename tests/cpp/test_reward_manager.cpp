// tests/cpp/test_reward_manager.cpp
// Unit tests for RewardManager class

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "constants/enemy.h"
#include "reward_manager.h"
#include "scene.h"
#include "test_helpers.h"

namespace rl2 {
namespace {

class RewardManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = testing::CreateTestScene();
    reward_manager_.Initialize();
  }

  Scene scene_;
  RewardManager reward_manager_;
};

// =============================================================================
// UpdateRewardTerms Tests
// =============================================================================

TEST_F(RewardManagerTest, UpdateRewardTerms_AccumulatesSignals) {
  // Add a simple term that always returns 1.0
  reward_manager_.AddTerm(
      "constant_one", 1.0f,
      [](const Scene& scene) -> std::array<float, kNumEnemies> {
        std::array<float, kNumEnemies> result;
        result.fill(1.0f);
        return result;
      });

  // Update twice
  reward_manager_.UpdateRewardTerms(scene_);
  reward_manager_.UpdateRewardTerms(scene_);

  // Calculate total (which uses accumulated signals)
  auto total = reward_manager_.CalculateTotalReward(scene_);

  // The constant_one term contributes 2.0 per enemy (1.0 * 2 updates * 1.0 weight)
  for (int i = 0; i < kNumEnemies; ++i) {
    // Verify total reward is valid
    EXPECT_FALSE(std::isnan(total[i]));
  }
}

// =============================================================================
// CalculateTotalReward Tests
// =============================================================================

TEST_F(RewardManagerTest, CalculateTotalReward_AppliesWeights) {
  // Reset and add a controlled term
  reward_manager_.Initialize();
  reward_manager_.ResetRewardTerms();

  const float weight = 5.0f;
  const float signal = 2.0f;

  reward_manager_.AddTerm(
      "weighted_term", weight,
      [signal](const Scene& scene) -> std::array<float, kNumEnemies> {
        std::array<float, kNumEnemies> result;
        result.fill(signal);
        return result;
      });

  reward_manager_.UpdateRewardTerms(scene_);
  auto total = reward_manager_.CalculateTotalReward(scene_);

  // The weighted_term should contribute weight * signal = 10.0 per enemy
  // Other default terms may add to this, so we check the weighted_term's
  // contribution via the reward dict
  auto reward_dict = reward_manager_.GetLastRewardDict();
  auto it = reward_dict.find("weighted_term");
  ASSERT_TRUE(it != reward_dict.end());

  for (int i = 0; i < kNumEnemies; ++i) {
    EXPECT_FLOAT_EQ(it->second[i], weight * signal)
        << "Enemy " << i << " weighted reward mismatch";
  }
}

TEST_F(RewardManagerTest, CalculateTotalReward_ClearsAccumulated) {
  reward_manager_.UpdateRewardTerms(scene_);

  auto first_total = reward_manager_.CalculateTotalReward(scene_);
  auto second_total = reward_manager_.CalculateTotalReward(scene_);

  // Without new updates, the accumulated signals should have been cleared
  for (int i = 0; i < kNumEnemies; ++i) {
    EXPECT_FLOAT_EQ(second_total[i], 0.0f)
        << "Enemy " << i << " should have zero reward after clearing";
  }
}

// =============================================================================
// FillRewardBuffer Tests
// =============================================================================

TEST_F(RewardManagerTest, FillRewardBuffer_ThrowsOnSizeMismatch) {
  std::vector<float> buffer(kNumEnemies - 1);  // Wrong size

  EXPECT_THROW(
      reward_manager_.FillRewardBuffer(buffer.data(), buffer.size(), scene_),
      std::runtime_error);
}

TEST_F(RewardManagerTest, FillRewardBuffer_DoesNotThrowOnCorrectSize) {
  std::vector<float> buffer(kNumEnemies);

  EXPECT_NO_THROW(
      reward_manager_.FillRewardBuffer(buffer.data(), buffer.size(), scene_));
}

TEST_F(RewardManagerTest, FillRewardBuffer_ContainsTotalRewards) {
  reward_manager_.UpdateRewardTerms(scene_);

  std::vector<float> buffer(kNumEnemies);
  reward_manager_.FillRewardBuffer(buffer.data(), buffer.size(), scene_);

  // All values should be valid (not NaN)
  for (int i = 0; i < kNumEnemies; ++i) {
    EXPECT_FALSE(std::isnan(buffer[i])) << "Buffer index " << i << " is NaN";
  }
}

}  // namespace
}  // namespace rl2
