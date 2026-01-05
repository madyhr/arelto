// src/rewards.cpp
#include <array>
#include "constants.h"
#include "reward_manager.h"
#include "scene.h"

namespace rl2 {

// This function is used to define all the reward terms.
void RewardManager::RegisterRewardTerms() {
  AddTerm(
      "velocity_towards_player", 1.0f,
      [](const Scene& scene) -> std::array<float, kNumEnemies> {
        std::array<float, kNumEnemies> value_array;

        for (int i = 0; i < kNumEnemies; ++i) {

          Vector2D dir_to_player =
              (scene.player.position_ - scene.enemy.position[i]).Normalized();
          float velocity_towards_player =
              scene.enemy.velocity[i].Dot(dir_to_player);
          value_array[i] = velocity_towards_player;
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
