#ifndef RL2_CONFIG_ENTITY_CONFIG_YAML_H_
#define RL2_CONFIG_ENTITY_CONFIG_YAML_H_

#include "config/config_decoding.h"
#include "config/entity_config.h"
#include "yaml-cpp/yaml.h"
namespace YAML {

using ::arelto::config::detail::DecodeMember;

template <>
struct convert<arelto::EnemyConfig> {
  static Node encode(const arelto::EnemyConfig& rhs) {
    Node node;
    node["max_health_points"] = rhs.max_health_points;
    node["attack_damage"] = rhs.attack_damage;
    node["attack_cooldown_s"] = rhs.attack_cooldown_s;
    node["movement_speed"] = rhs.movement_speed;
    node["width"] = rhs.width;
    node["height"] = rhs.height;
    node["inv_mass"] = rhs.inv_mass;
    return node;
  }

  static bool decode(const Node& node, arelto::EnemyConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "entity.enemy";

    DecodeMember(node, "max_health_points", rhs,
                 &arelto::EnemyConfig::max_health_points, kOwner);
    DecodeMember(node, "attack_damage", rhs,
                 &arelto::EnemyConfig::attack_damage, kOwner);
    DecodeMember(node, "attack_cooldown_s", rhs,
                 &arelto::EnemyConfig::attack_cooldown_s, kOwner);
    DecodeMember(node, "movement_speed", rhs,
                 &arelto::EnemyConfig::movement_speed, kOwner);
    DecodeMember(node, "width", rhs, &arelto::EnemyConfig::width, kOwner);
    DecodeMember(node, "height", rhs, &arelto::EnemyConfig::height, kOwner);
    DecodeMember(node, "inv_mass", rhs, &arelto::EnemyConfig::inv_mass, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::EntityConfig> {
  static Node encode(const arelto::EntityConfig& rhs) {
    Node node;
    node["enemy"] = rhs.enemy;
    return node;
  }

  static bool decode(const Node& node, arelto::EntityConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "entity";

    DecodeMember(node, "enemy", rhs, &arelto::EntityConfig::enemy, kOwner);
    return true;
  }
};

}  // namespace YAML

#endif
