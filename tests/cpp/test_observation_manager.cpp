// tests/cpp/test_observation_manager.cpp
// Unit tests for ObservationManager class

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "constants/enemy.h"
#include "constants/ray_caster.h"
#include "observation_manager.h"
#include "scene.h"
#include "test_helpers.h"

namespace arelto {
namespace {

class ObservationManagerTest : public ::testing::Test {
 protected:
  void SetUp() override { scene_ = testing::CreateTestScene(); }

  Scene scene_;
  ObservationManager obs_manager_;
};

// =============================================================================
// FillObservationBuffer Tests
// =============================================================================

TEST_F(ObservationManagerTest,
       FillObservationBuffer_ThrowsOnSizeMismatch_TooSmall) {
  int size = obs_manager_.GetObservationSize(scene_);
  std::vector<float> buffer(kNumEnemies * size - 1);  // One less than expected

  EXPECT_THROW(
      obs_manager_.FillObservationBuffer(buffer.data(), buffer.size(), scene_),
      std::runtime_error);
}

TEST_F(ObservationManagerTest,
       FillObservationBuffer_ThrowsOnSizeMismatch_TooLarge) {
  int size = obs_manager_.GetObservationSize(scene_);
  std::vector<float> buffer(kNumEnemies * size + 1);  // One more than expected

  EXPECT_THROW(
      obs_manager_.FillObservationBuffer(buffer.data(), buffer.size(), scene_),
      std::runtime_error);
}

TEST_F(ObservationManagerTest,
       FillObservationBuffer_DoesNotThrowOnCorrectSize) {
  int size = obs_manager_.GetObservationSize(scene_);
  std::vector<float> buffer(kNumEnemies * size);

  EXPECT_NO_THROW(
      obs_manager_.FillObservationBuffer(buffer.data(), buffer.size(), scene_));
}

TEST_F(ObservationManagerTest, FillObservationBuffer_ContainsTypeInformation) {
  // Set known entity types in the ray hit data
  // We'll set a specific ray/enemy combo to a specific type to verify exact placement
  int history_idx = scene_.enemy.ray_caster.history_idx;
  int target_ray = 0;
  int target_enemy = 0;
  EntityType target_type = EntityType::player;

  scene_.enemy.ray_caster.ray_hit_types[history_idx][target_ray][target_enemy] =
      target_type;

  int size = obs_manager_.GetObservationSize(scene_);
  std::vector<float> buffer(kNumEnemies * size);
  obs_manager_.FillObservationBuffer(buffer.data(), buffer.size(), scene_);
  bool found_specific_type = false;
  for (float val : buffer) {
    if (std::abs(val - static_cast<float>(target_type)) < 1e-5) {
      found_specific_type = true;
      break;
    }
  }
  EXPECT_TRUE(found_specific_type)
      << "Did not find the specific entity type value in buffer";
}

TEST_F(ObservationManagerTest, FillObservationBuffer_BufferFullyPopulated) {
  // Initialize buffer with NaN to detect unpopulated values
  int size = obs_manager_.GetObservationSize(scene_);
  std::vector<float> buffer(kNumEnemies * size,
                            std::numeric_limits<float>::quiet_NaN());

  obs_manager_.FillObservationBuffer(buffer.data(), buffer.size(), scene_);

  // All values should be populated (not NaN)
  for (size_t i = 0; i < buffer.size(); ++i) {
    EXPECT_FALSE(std::isnan(buffer[i])) << "Buffer index " << i << " is NaN";
  }
}

}  // namespace
}  // namespace arelto
