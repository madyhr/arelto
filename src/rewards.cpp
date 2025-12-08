// src/rewards.cpp
#include <array>
#include "constants.h"
#include "reward_manager.h"
#include "scene.h"

namespace rl2 {

// This function is used to define all the reward terms.
void RewardManager::RegisterRewardTerms() {

  AddTerm("staying_alive", 1.0f,
          [](const Scene& scene) -> std::array<float, kNumEnemies> {
            std::array<float, kNumEnemies> value_array;
            for (int i = 0; i < kNumEnemies; ++i) {
              value_array[i] =
                  static_cast<int>(scene.enemy.is_alive[i]) * kPhysicsDt;
            }
            return value_array;
          });

  AddTerm("damage_dealt", 1.0f,
          [](const Scene& scene) -> std::array<float, kNumEnemies> {
            std::array<float, kNumEnemies> value_array;
            for (int i = 0; i < kNumEnemies; ++i) {
              value_array[i] =
                  static_cast<float>(scene.enemy.damage_dealt_sim_step[i]);
            }
            return value_array;
          });
}

}  // namespace rl2
