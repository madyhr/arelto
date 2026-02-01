// src/rewards.cpp
#include <array>
#include "constants/enemy.h"
#include "constants/ray_caster.h"
#include "ray_caster.h"
#include "reward_manager.h"
#include "scene.h"

namespace arelto {

// This function is used to define all the reward terms.
void RewardManager::RegisterRewardTerms() {
  AddTerm(
      "velocity_towards_player", 1.0f,
      [](const Scene& scene) -> std::array<float, kNumEnemies> {
        std::array<float, kNumEnemies> value_array;

        int current_history_idx =
            (scene.enemy.ray_caster.history_idx - 1 + kRayHistoryLength) %
            kRayHistoryLength;
        for (int i = 0; i < kNumEnemies; ++i) {

          Vector2D dir_to_player =
              (scene.player.position_ - scene.enemy.position[i]).Normalized();
          float velocity_towards_player =
              scene.enemy.velocity[i].Dot(dir_to_player);

          bool is_player_detected =
              IsEntityTypePresent(scene.enemy.ray_caster.ray_hit_types,
                                  current_history_idx, i, EntityType::player);
          value_array[i] = velocity_towards_player * is_player_detected;
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

}  // namespace arelto
