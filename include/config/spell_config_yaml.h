#ifndef RL2_CONFIG_SPELL_CONFIG_YAML_H_
#define RL2_CONFIG_SPELL_CONFIG_YAML_H_

#include "config/config_decoding.h"
#include "config/spell_config.h"
#include "types.h"
#include "yaml-cpp/yaml.h"

namespace YAML {

template <>
struct convert<arelto::SpellConfig> {
  static Node encode(const arelto::SpellConfig& rhs) {
    Node node;
    node["name"] = rhs.name;
    node["width"] = rhs.width;
    node["height"] = rhs.height;
    node["speed"] = rhs.speed;
    node["damage"] = rhs.damage;
    node["cooldown"] = rhs.cooldown;
    return node;
  }

  static bool decode(const Node& node, arelto::SpellConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "spells";

    using arelto::CreateCenteredCollider;
    using arelto::config::detail::DecodeMember;

    DecodeMember(node, "name", rhs, &arelto::SpellConfig::name, kOwner);
    DecodeMember(node, "width", rhs, &arelto::SpellConfig::width, kOwner);
    DecodeMember(node, "height", rhs, &arelto::SpellConfig::height, kOwner);
    DecodeMember(node, "sprite_cell_width", rhs,
                 &arelto::SpellConfig::sprite_cell_width, kOwner);
    DecodeMember(node, "sprite_cell_height", rhs,
                 &arelto::SpellConfig::sprite_cell_height, kOwner);
    DecodeMember(node, "speed", rhs, &arelto::SpellConfig::speed, kOwner);
    DecodeMember(node, "damage", rhs, &arelto::SpellConfig::damage, kOwner);
   DecodeMember(node, "cooldown", rhs, &arelto::SpellConfig::cooldown, kOwner);

    rhs.collider = CreateCenteredCollider({rhs.width, rhs.height});

    return true;
  }
};

}  // namespace YAML

#endif
