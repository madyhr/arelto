#ifndef RL2_CONFIG_ITEM_CONFIG_YAML_H_
#define RL2_CONFIG_ITEM_CONFIG_YAML_H_

#include <array>
#include "config/config_decoding.h"
#include "config/item_config.h"
#include "yaml-cpp/yaml.h"

namespace YAML {

using ::arelto::config::detail::DecodeMember;

template <>
struct convert<arelto::ItemDefinitionConfig> {
  static Node encode(const arelto::ItemDefinitionConfig& rhs) {
    Node node;
    node["flavor_text"] = rhs.flavor_text;
    return node;
  }

  static bool decode(const Node& node, arelto::ItemDefinitionConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "items.item";

    DecodeMember(node, "flavor_text", rhs,
                 &arelto::ItemDefinitionConfig::flavor_text, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::ItemConfig> {
  static Node encode(const arelto::ItemConfig& rhs) {
    Node node;
    for (const ItemMapping& mapping : kItemMappings) {
      node["items"][mapping.key] = rhs.items[to_index(mapping.item_id)];
    }
    return node;
  }

  static bool decode(const Node& node, arelto::ItemConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    const Node items = node["items"];
    if (!items) {
      return true;
    }
    if (!items.IsMap()) {
      return false;
    }

    for (const ItemMapping& mapping : kItemMappings) {
      if (!DecodeItem(items, mapping.key, mapping.item_id, rhs)) {
        return false;
      }
    }
    return true;
  }

 private:
  struct ItemMapping {
    const char* key;
    arelto::ItemId item_id;
  };

  inline static constexpr std::array<ItemMapping,
                                     to_index(arelto::ItemId::count)>
      kItemMappings{{
          {"elia_armor_plate", arelto::ItemId::elia_armor_plate},
          {"damodei_claw", arelto::ItemId::damodei_claw},
          {"volmnih_boots", arelto::ItemId::volmnih_boots},
          {"sarto_button_bible", arelto::ItemId::sarto_button_bible},
          {"aiayn_scale", arelto::ItemId::aiayn_scale},
      }};

  static bool DecodeItem(const Node& items, const char* key,
                         arelto::ItemId item_id, arelto::ItemConfig& rhs) {
    const Node item_node = items[key];
    if (!item_node) {
      return true;
    }
    return convert<arelto::ItemDefinitionConfig>::decode(
        item_node, rhs.items[to_index(item_id)]);
  }
};

}  // namespace YAML

#endif
