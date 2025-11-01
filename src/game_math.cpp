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

Vector2D get_centroid(Vector2D position, Size size) {
  return {position.x + 0.5f * size.width, position.y + 0.5f * size.height};
}

void handle_collisions_sap(Player& player, Enemy& enemy) {
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

  std::vector<CollisionPair> collision_pairs =
    get_collision_pairs_sap(sorted_aabb);
  resolve_collision_pairs_sap(player, enemy, entity_aabb, collision_pairs);
};

std::vector<CollisionPair> get_collision_pairs_sap(
  std::array<AABB, kNumEntities> sorted_aabb) {
  std::vector<CollisionPair> collision_pairs;
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

  return collision_pairs;
};

void resolve_collision_pairs_sap(Player& player, Enemy& enemy,
                                 std::array<AABB, kNumEntities> entity_aabb,
                                 std::vector<CollisionPair> collision_pairs) {
  for (const CollisionPair& cp : collision_pairs) {
    const AABB& a = entity_aabb[cp.index_a];
    const AABB& b = entity_aabb[cp.index_b];

    float overlap_x = std::min(a.max_x, b.max_x) - std::max(a.min_x, b.min_x);
    float overlap_y = std::min(a.max_y, b.max_y) - std::max(a.min_y, b.min_y);

    if (overlap_x <= 0.0f || overlap_y <= 0.0f)
      continue;  // No overlap

    // Choose smaller axis
    bool resolve_x = (overlap_x < overlap_y);

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
    auto get_entity_centroid = [&](int idx) -> Vector2D {
      if (idx == 0) {
        return rl2::get_centroid(player.position, player.stats.size);
      } else {
        int enemy_idx = idx - 1;
        return rl2::get_centroid(enemy.position[enemy_idx],
                                 enemy.size[enemy_idx]);
      }
    };
    auto get_entity_inv_mass = [&](int idx) -> float {
      if (idx == 0) {
        return player.stats.inv_mass;
      } else {
        int enemy_idx = idx - 1;
        return enemy.inv_mass[enemy_idx];
      }
    };
    float inv_mass_a = get_entity_inv_mass(cp.index_a);
    float inv_mass_b = get_entity_inv_mass(cp.index_b);
    float push_factor = inv_mass_b / (inv_mass_a + inv_mass_b);

    if (resolve_x) {
      Vector2D centroid_a = get_entity_centroid(cp.index_a);
      Vector2D centroid_b = get_entity_centroid(cp.index_b);
      // Determine the separation direction: 1.0f if B is to the right of A, -1.0f otherwise
      float direction = (centroid_b.x - centroid_a.x >= 0.0f) ? 1.0f : -1.0f;

      move_entity(cp.index_a, -direction * overlap_x * (1.0f - push_factor),
                  0.0f);
      move_entity(cp.index_b, direction * overlap_x * push_factor, 0.0f);
    } else {
      Vector2D centroid_a = get_entity_centroid(cp.index_a);
      Vector2D centroid_b = get_entity_centroid(cp.index_b);
      // Determine the separation direction: 1.0f if B is to the right of A, -1.0f otherwise
      float direction = (centroid_b.y - centroid_a.y >= 0.0f) ? 1.0f : -1.0f;

      move_entity(cp.index_a, 0.0f,
                  -direction * overlap_y * (1.0f - push_factor));
      move_entity(cp.index_b, 0.0f, direction * overlap_y * push_factor);
    }
  }
};

void handle_player_oob(Player& player) {
  if (player.position.x < 0) {
    player.position.x = 0;
  }
  if (player.position.y < 0) {
    player.position.y = 0;
  }
  if ((player.position.x + player.stats.size.width) > kMapWidth) {
    player.position.x = kMapWidth - player.stats.size.width;
  }
  if ((player.position.y + player.stats.size.height) > kMapHeight) {
    player.position.y = kMapHeight - player.stats.size.height;
  }
};

void handle_enemy_oob(Enemy& enemy) {
  for (int i = 0; i < kNumEnemies; ++i) {
    if (enemy.is_alive[i]) {
      if (enemy.position[i].x < 0) {
        enemy.position[i].x = 0;
      }
      if (enemy.position[i].y < 0) {
        enemy.position[i].y = 0;
      }
      if ((enemy.position[i].x + enemy.size[i].width) > kMapWidth) {
        enemy.position[i].x = kMapWidth - enemy.size[i].width;
      }
      if ((enemy.position[i].y + enemy.size[i].height) > kMapHeight) {
        enemy.position[i].y = kMapHeight - enemy.size[i].height;
      }
    }
  }
};

}  // namespace rl2
