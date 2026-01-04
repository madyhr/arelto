// src/rewards.cpp
#include <array>
#include "constants.h"
#include "reward_manager.h"
#include "scene.h"

namespace rl2 {

// This function is used to define all the reward terms.
void RewardManager::RegisterRewardTerms() {

  AddTerm("distance_progress", 1.0f,
          [](const Scene& scene) -> std::array<float, kNumEnemies> {
            std::array<float, kNumEnemies> value_array;

            for (int i = 0; i < kNumEnemies; ++i) {
              float prev_distance_to_player =
                  (scene.player.prev_position_ - scene.enemy.prev_position[i])
                      .Norm();
              float distance_to_player =
                  (scene.player.position_ - scene.enemy.position[i]).Norm();
              value_array[i] = prev_distance_to_player - distance_to_player;
            }

            return value_array;
          });

  AddTerm("distance_to_player_exp", 1.0f,
          [](const Scene& scene) -> std::array<float, kNumEnemies> {
            std::array<float, kNumEnemies> value_array;

            for (int i = 0; i < kNumEnemies; ++i) {
              float distance_to_player =
                  (scene.player.position_ - scene.enemy.position[i]).Norm();
              // 0.005 is the inv scale of the exponential kernel. It is defined
              // s.t. it has a value of 1/e at 200 units of distance. Used to
              // incentivize being close to the player.
              value_array[i] = std::exp(-distance_to_player * 0.005);
            }

            return value_array;
          });

  AddTerm("was_terminated", -1000.0f,
          [](const Scene& scene) -> std::array<float, kNumEnemies> {
            std::array<float, kNumEnemies> value_array;
            for (int i = 0; i < kNumEnemies; ++i) {
              value_array[i] =
                  static_cast<float>(scene.enemy.is_terminated_latched[i]);
            }
            return value_array;
          });

  AddTerm("damage_dealt", 100.0f,
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
