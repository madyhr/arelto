// src/rewards.cpp
#include <array>
#include "constants.h"
#include "reward_manager.h"
#include "scene.h"

namespace rl2 {

// This function is used to define all the reward terms.
void RewardManager::RegisterRewardTerms() {

  AddTerm("proximity_to_player", 0.01f,
          [](const Scene& scene) -> std::array<float, kNumEnemies> {
            std::array<float, kNumEnemies> value_array;

            for (int i = 0; i < kNumEnemies; ++i) {
              float distance_to_player =
                  (scene.player.position_ - scene.enemy.position[i]).Norm();
              value_array[i] = -(distance_to_player * kInvMapMaxDistance);
            }

            return value_array;
          });

  // AddTerm("staying_alive", 1.0f,
  //         [](const Scene& scene) -> std::array<float, kNumEnemies> {
  //           std::array<float, kNumEnemies> value_array;
  //           for (int i = 0; i < kNumEnemies; ++i) {
  //             value_array[i] =
  //                 static_cast<float>(scene.enemy.is_alive[i]) * kPhysicsDt;
  //           }
  //           return value_array;
  //         });

  AddTerm("damage_dealt", 10.0f,
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
