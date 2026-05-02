#ifndef RL2_CONFIG_ENTITY_CONFIG_YAML_H_
#define RL2_CONFIG_ENTITY_CONFIG_YAML_H_

#include "config/config_decoding.h"
#include "config/entity_config.h"
#include "yaml-cpp/yaml.h"
namespace YAML {

using ::arelto::config::detail::DecodeMember;

template <>
struct convert<arelto::PlayerConfig> {
  static Node encode(const arelto::PlayerConfig& rhs) {
    Node node;
    node["max_health_points"] = rhs.max_health_points;
    node["spawn_x"] = rhs.spawn_x;
    node["spawn_y"] = rhs.spawn_y;
    node["movement_speed"] = rhs.movement_speed;
    node["width"] = rhs.width;
    node["height"] = rhs.height;
    node["inv_mass"] = rhs.inv_mass;
    node["initial_exp_requirement"] = rhs.initial_exp_requirement;
    node["exp_required_scale"] = rhs.exp_required_scale;
    node["invulnerable_window_s"] = rhs.invulnerable_window_s;
    return node;
  }

  static bool decode(const Node& node, arelto::PlayerConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "entity.player";

    DecodeMember(node, "max_health_points", rhs,
                 &arelto::PlayerConfig::max_health_points, kOwner);
    DecodeMember(node, "spawn_x", rhs, &arelto::PlayerConfig::spawn_x, kOwner);
    DecodeMember(node, "spawn_y", rhs, &arelto::PlayerConfig::spawn_y, kOwner);
    DecodeMember(node, "movement_speed", rhs,
                 &arelto::PlayerConfig::movement_speed, kOwner);
    DecodeMember(node, "width", rhs, &arelto::PlayerConfig::width, kOwner);
    DecodeMember(node, "height", rhs, &arelto::PlayerConfig::height, kOwner);
    DecodeMember(node, "inv_mass", rhs, &arelto::PlayerConfig::inv_mass,
                 kOwner);
    DecodeMember(node, "initial_exp_requirement", rhs,
                 &arelto::PlayerConfig::initial_exp_requirement, kOwner);
    DecodeMember(node, "exp_required_scale", rhs,
                 &arelto::PlayerConfig::exp_required_scale, kOwner);
    DecodeMember(node, "invulnerable_window_s", rhs,
                 &arelto::PlayerConfig::invulnerable_window_s, kOwner);
    return true;
  }
};

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
struct convert<arelto::ExpGemRarityConfig> {
  static Node encode(const arelto::ExpGemRarityConfig& rhs) {
    Node node;
    node["exp_value"] = rhs.exp_value;
    node["width"] = rhs.width;
    node["height"] = rhs.height;
    return node;
  }

  static bool decode(const Node& node, arelto::ExpGemRarityConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "entity.exp_gem.rarity";

    DecodeMember(node, "exp_value", rhs, &arelto::ExpGemRarityConfig::exp_value,
                 kOwner);
    DecodeMember(node, "width", rhs, &arelto::ExpGemRarityConfig::width,
                 kOwner);
    DecodeMember(node, "height", rhs, &arelto::ExpGemRarityConfig::height,
                 kOwner);
    return true;
  }
};

template <>
struct convert<arelto::ExpGemConfig> {
  static Node encode(const arelto::ExpGemConfig& rhs) {
    Node node;
    node["inv_mass"] = rhs.inv_mass;
    node["rarities"]["common"] = rhs.rarities[arelto::Rarity::common];
    node["rarities"]["rare"] = rhs.rarities[arelto::Rarity::rare];
    node["rarities"]["epic"] = rhs.rarities[arelto::Rarity::epic];
    node["rarities"]["legendary"] = rhs.rarities[arelto::Rarity::legendary];
    return node;
  }

  static bool decode(const Node& node, arelto::ExpGemConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "entity.exp_gem";

    DecodeMember(node, "inv_mass", rhs, &arelto::ExpGemConfig::inv_mass,
                 kOwner);

    const Node rarities = node["rarities"];
    if (!rarities) {
      return true;
    }
    if (!rarities.IsMap()) {
      return false;
    }

    return DecodeRarity(rarities, "common", arelto::Rarity::common, rhs) &&
           DecodeRarity(rarities, "rare", arelto::Rarity::rare, rhs) &&
           DecodeRarity(rarities, "epic", arelto::Rarity::epic, rhs) &&
           DecodeRarity(rarities, "legendary", arelto::Rarity::legendary, rhs);
  }

 private:
  static bool DecodeRarity(const Node& rarities, const char* key,
                           arelto::Rarity rarity, arelto::ExpGemConfig& rhs) {
    const Node rarity_node = rarities[key];
    if (!rarity_node) {
      return true;
    }
    return convert<arelto::ExpGemRarityConfig>::decode(rarity_node,
                                                       rhs.rarities[rarity]);
  }
};

template <>
struct convert<arelto::EntityConfig> {
  static Node encode(const arelto::EntityConfig& rhs) {
    Node node;
    node["player"] = rhs.player;
    node["enemy"] = rhs.enemy;
    node["exp_gem"] = rhs.exp_gem;
    return node;
  }

  static bool decode(const Node& node, arelto::EntityConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "entity";

    DecodeMember(node, "player", rhs, &arelto::EntityConfig::player, kOwner);
    DecodeMember(node, "enemy", rhs, &arelto::EntityConfig::enemy, kOwner);
    DecodeMember(node, "exp_gem", rhs, &arelto::EntityConfig::exp_gem, kOwner);
    return true;
  }
};

}  // namespace YAML

#endif
