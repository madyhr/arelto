// src/game_math.cpp
#include "game_math.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include "constants.h"
#include "entity.h"
#include "types.h"

namespace rl2 {

float get_length_vector2d(Vector2D vector) {
  float length = std::hypot(vector.x, vector.y);
  return length;
};

float calculate_distance_vector2d(Vector2D v0, Vector2D v1) {
  float dx = v1.x - v0.x;
  float dy = v1.y - v0.y;
  float distance = std::hypot(dx, dy);
  return distance;
};

void resolve_collisions_sap(Player& player, Enemy& enemy) {
  std::vector<CollisionPair> collision_pairs;

  std::array<AABB, kNumEntities> entity_aabb;
  player.update_aabb();
  entity_aabb[0] = player.aabb;
  for (int i = 1; i < kNumEnemies + 1; ++i) {
    entity_aabb[i] = {enemy.position[i - 1].x, enemy.position[i - 1].y,
                      enemy.position[i - 1].x + enemy.size[i - 1].width,
                      enemy.position[i - 1].y + enemy.size[i - 1].height, i};
  }

  std::array<AABB, kNumEntities> sorted_aabb = entity_aabb;
  std::sort(sorted_aabb.begin(), sorted_aabb.end(),
            [](const AABB& a, const AABB& b) { return a.min_x < b.min_x; });
  std::vector<const AABB*> active_list;
  for (int i = 0; i < kNumEntities; ++i) {
    const AABB& current_aabb = sorted_aabb[i];

    // Prune
    active_list.erase(std::remove_if(active_list.begin(), active_list.end(),
                                     [&](const AABB* active_aabb) {
                                       return active_aabb->max_x <
                                              current_aabb.min_x;
                                     }),
                      active_list.end());
    // Search
    for (const AABB* active_aabb : active_list) {
      bool has_y_overlap = current_aabb.max_y > active_aabb->min_y &&
                           current_aabb.min_y < active_aabb->max_y;
      if (has_y_overlap) {

        collision_pairs.push_back(
            {current_aabb.entity_idx, active_aabb->entity_idx});
      };
    }

    // Add
    active_list.push_back(&current_aabb);
  }
  // --- Narrowphase and resolution ---
  for (const CollisionPair& cp : collision_pairs) {
    const AABB& a = entity_aabb[cp.index_a];
    const AABB& b = entity_aabb[cp.index_b];

    float overlap_x = std::min(a.max_x, b.max_x) - std::max(a.min_x, b.min_x);
    float overlap_y = std::min(a.max_y, b.max_y) - std::max(a.min_y, b.min_y);

    if (overlap_x <= 0.0f || overlap_y <= 0.0f)
      continue;  // No overlap

    // Choose smaller axis
    bool resolve_x = (overlap_x < overlap_y);
    const float push_factor = 0.5f;  // move both halves equally

    auto move_entity = [&](int idx, float dx, float dy) {
      if (idx == 0) {
        player.position.x += dx;
        player.position.y += dy;
      } else {
        int enemy_idx = idx - 1;
        enemy.position[enemy_idx].x += dx;
        enemy.position[enemy_idx].y += dy;
      }
    };
    if (resolve_x) {
      // Get current, live position/center for direction calculation
      // Helper lambda to get the center of a live entity based on its index
      auto get_center_x = [&](int idx, const AABB& box) -> float {
        if (idx == 0) {
          return player.position.x + player.stats.size.width * 0.5f;
        } else {
          int enemy_idx = idx - 1;
          return enemy.position[enemy_idx].x +
                 enemy.size[enemy_idx].width * 0.5f;
        }
      };

      float center_a_x = get_center_x(cp.index_a, a);
      float center_b_x = get_center_x(cp.index_b, b);

      // Determine the separation direction: 1.0f if B is to the right of A, -1.0f otherwise
      float direction = (center_b_x - center_a_x >= 0.0f) ? 1.0f : -1.0f;

      move_entity(cp.index_a, -direction * overlap_x * push_factor, 0.0f);
      move_entity(cp.index_b, direction * overlap_x * push_factor, 0.0f);
    } else {
      // Helper lambda to get the center of a live entity based on its index
      auto get_center_y = [&](int idx, const AABB& box) -> float {
        if (idx == 0) {
          return player.position.y + player.stats.size.height * 0.5f;
        } else {
          int enemy_idx = idx - 1;
          return enemy.position[enemy_idx].y +
                 enemy.size[enemy_idx].height * 0.5f;
        }
      };

      float center_a_y = get_center_y(cp.index_a, a);
      float center_b_y = get_center_y(cp.index_b, b);

      float direction = (center_b_y - center_a_y >= 0.0f) ? 1.0f : -1.0f;

      move_entity(cp.index_a, 0.0f, -direction * overlap_y * push_factor);
      move_entity(cp.index_b, 0.0f, direction * overlap_y * push_factor);
    }
  }
}

}  // namespace rl2
