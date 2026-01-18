// tests/cpp/test_action_manager.cpp
// Unit tests for ActionManager class

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "action_manager.h"
#include "constants/enemy.h"
#include "scene.h"
#include "test_helpers.h"

namespace rl2 {
namespace {

class ActionManagerTest : public ::testing::Test {
 protected:
  void SetUp() override { scene_ = testing::CreateTestScene(); }

  Scene scene_;
  ActionManager action_manager_;
};

// =============================================================================
// GetActionSize Tests
// =============================================================================

TEST_F(ActionManagerTest, GetActionSize_ReturnsTwo) {
  // Action size should be 2 (x velocity and y velocity)
  int action_size = action_manager_.GetActionSize(scene_);
  EXPECT_EQ(action_size, 2);
}

// =============================================================================
// ReadActionBuffer Tests
// =============================================================================

TEST_F(ActionManagerTest, ReadActionBuffer_SetsVelocitiesCorrectly_AllZero) {
  // Buffer value 0 -> velocity -1.0f
  // Buffer value 1 -> velocity 0.0f
  // Buffer value 2 -> velocity 1.0f

  int action_size = action_manager_.GetActionSize(scene_);
  std::vector<int> buffer(kNumEnemies * action_size);

  // Set all actions to 1 (which maps to 0.0f velocity)
  std::fill(buffer.begin(), buffer.end(), 1);

  action_manager_.ReadActionBuffer(buffer.data(), buffer.size(), scene_);

  for (int i = 0; i < kNumEnemies; ++i) {
    EXPECT_FLOAT_EQ(scene_.enemy.velocity[i].x, 0.0f)
        << "Enemy " << i << " x velocity mismatch";
    EXPECT_FLOAT_EQ(scene_.enemy.velocity[i].y, 0.0f)
        << "Enemy " << i << " y velocity mismatch";
  }
}

TEST_F(ActionManagerTest,
       ReadActionBuffer_SetsVelocitiesCorrectly_AllPositive) {
  int action_size = action_manager_.GetActionSize(scene_);
  std::vector<int> buffer(kNumEnemies * action_size);

  // Set all x velocities to 2 (-> 1.0f) and all y velocities to 2 (-> 1.0f)
  std::fill(buffer.begin(), buffer.begin() + kNumEnemies, 2);
  std::fill(buffer.begin() + kNumEnemies, buffer.end(), 2);

  action_manager_.ReadActionBuffer(buffer.data(), buffer.size(), scene_);

  for (int i = 0; i < kNumEnemies; ++i) {
    EXPECT_FLOAT_EQ(scene_.enemy.velocity[i].x, 1.0f)
        << "Enemy " << i << " x velocity mismatch";
    EXPECT_FLOAT_EQ(scene_.enemy.velocity[i].y, 1.0f)
        << "Enemy " << i << " y velocity mismatch";
  }
}

TEST_F(ActionManagerTest,
       ReadActionBuffer_SetsVelocitiesCorrectly_AllNegative) {
  int action_size = action_manager_.GetActionSize(scene_);
  std::vector<int> buffer(kNumEnemies * action_size);

  // Set all x velocities to 0 (-> -1.0f) and all y velocities to 0 (-> -1.0f)
  std::fill(buffer.begin(), buffer.end(), 0);

  action_manager_.ReadActionBuffer(buffer.data(), buffer.size(), scene_);

  for (int i = 0; i < kNumEnemies; ++i) {
    EXPECT_FLOAT_EQ(scene_.enemy.velocity[i].x, -1.0f)
        << "Enemy " << i << " x velocity mismatch";
    EXPECT_FLOAT_EQ(scene_.enemy.velocity[i].y, -1.0f)
        << "Enemy " << i << " y velocity mismatch";
  }
}

TEST_F(ActionManagerTest, ReadActionBuffer_SetsVelocitiesCorrectly_Mixed) {
  int action_size = action_manager_.GetActionSize(scene_);
  std::vector<int> buffer(kNumEnemies * action_size);

  // Set x velocities to 2 (-> 1.0f) and y velocities to 0 (-> -1.0f)
  std::fill(buffer.begin(), buffer.begin() + kNumEnemies, 2);
  std::fill(buffer.begin() + kNumEnemies, buffer.end(), 0);

  action_manager_.ReadActionBuffer(buffer.data(), buffer.size(), scene_);

  for (int i = 0; i < kNumEnemies; ++i) {
    EXPECT_FLOAT_EQ(scene_.enemy.velocity[i].x, 1.0f)
        << "Enemy " << i << " x velocity mismatch";
    EXPECT_FLOAT_EQ(scene_.enemy.velocity[i].y, -1.0f)
        << "Enemy " << i << " y velocity mismatch";
  }
}

TEST_F(ActionManagerTest, ReadActionBuffer_ThrowsOnSizeMismatch_TooSmall) {
  int action_size = action_manager_.GetActionSize(scene_);
  std::vector<int> buffer(kNumEnemies * action_size -
                          1);  // One less than expected

  EXPECT_THROW(
      action_manager_.ReadActionBuffer(buffer.data(), buffer.size(), scene_),
      std::runtime_error);
}

TEST_F(ActionManagerTest, ReadActionBuffer_ThrowsOnSizeMismatch_TooLarge) {
  int action_size = action_manager_.GetActionSize(scene_);
  std::vector<int> buffer(kNumEnemies * action_size +
                          1);  // One more than expected

  EXPECT_THROW(
      action_manager_.ReadActionBuffer(buffer.data(), buffer.size(), scene_),
      std::runtime_error);
}

TEST_F(ActionManagerTest, ReadActionBuffer_AllEnemiesReceiveActions) {
  int action_size = action_manager_.GetActionSize(scene_);
  std::vector<int> buffer(kNumEnemies * action_size);

  // Set unique actions for each enemy to verify proper distribution
  for (int i = 0; i < kNumEnemies; ++i) {
    buffer[i] = i % 3;                      // x velocity: cycles 0,1,2
    buffer[kNumEnemies + i] = (i + 1) % 3;  // y velocity: cycles 1,2,0
  }

  action_manager_.ReadActionBuffer(buffer.data(), buffer.size(), scene_);

  for (int i = 0; i < kNumEnemies; ++i) {
    float expected_vx = static_cast<float>((i % 3) - 1);
    float expected_vy = static_cast<float>(((i + 1) % 3) - 1);

    EXPECT_FLOAT_EQ(scene_.enemy.velocity[i].x, expected_vx)
        << "Enemy " << i << " x velocity mismatch";
    EXPECT_FLOAT_EQ(scene_.enemy.velocity[i].y, expected_vy)
        << "Enemy " << i << " y velocity mismatch";
  }
}

}  // namespace
}  // namespace rl2
