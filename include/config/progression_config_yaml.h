#ifndef RL2_CONFIG_PROGRESSION_CONFIG_YAML_H_
#define RL2_CONFIG_PROGRESSION_CONFIG_YAML_H_

#include "config/config_decoding.h"
#include "config/progression_config.h"
#include "yaml-cpp/yaml.h"

namespace YAML {

using ::arelto::config::detail::DecodeField;
using ::arelto::config::detail::DecodeMember;

template <>
struct convert<arelto::SpellUpgradeConfig> {
  static Node encode(const arelto::SpellUpgradeConfig& rhs) {
    Node node;
    node["rarity_weights"]["common"] =
        rhs.rarity_weights[to_index(arelto::Rarity::common)];
    node["rarity_weights"]["rare"] =
        rhs.rarity_weights[to_index(arelto::Rarity::rare)];
    node["rarity_weights"]["epic"] =
        rhs.rarity_weights[to_index(arelto::Rarity::epic)];
    node["rarity_weights"]["legendary"] =
        rhs.rarity_weights[to_index(arelto::Rarity::legendary)];
    return node;
  }

  static bool decode(const Node& node, arelto::SpellUpgradeConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    const Node rarity_weights = node["rarity_weights"];
    if (!rarity_weights) {
      return true;
    }
    if (!rarity_weights.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "progression.spell_upgrade.rarity_weights";

    DecodeRarityWeight(rarity_weights, "common", arelto::Rarity::common, rhs,
                       kOwner);
    DecodeRarityWeight(rarity_weights, "rare", arelto::Rarity::rare, rhs,
                       kOwner);
    DecodeRarityWeight(rarity_weights, "epic", arelto::Rarity::epic, rhs,
                       kOwner);
    DecodeRarityWeight(rarity_weights, "legendary", arelto::Rarity::legendary,
                       rhs, kOwner);
    return true;
  }

 private:
  static void DecodeRarityWeight(const Node& rarity_weights, const char* key,
                                 arelto::Rarity rarity,
                                 arelto::SpellUpgradeConfig& rhs,
                                 const char* owner) {
    DecodeField(rarity_weights, key, rhs.rarity_weights[to_index(rarity)],
                owner);
  }
};

template <>
struct convert<arelto::ProgressionConfig> {
  static Node encode(const arelto::ProgressionConfig& rhs) {
    Node node;
    node["spell_upgrade"] = rhs.spell_upgrade;
    return node;
  }

  static bool decode(const Node& node, arelto::ProgressionConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "progression";

    DecodeMember(node, "spell_upgrade", rhs,
                 &arelto::ProgressionConfig::spell_upgrade, kOwner);
    return true;
  }
};

}  // namespace YAML

#endif
