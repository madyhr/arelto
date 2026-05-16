#ifndef RL2_CONFIG_ITEM_CONFIG_H_
#define RL2_CONFIG_ITEM_CONFIG_H_

#include <array>
#include <string>
#include "items.h"

namespace arelto {

struct ItemDefinitionConfig {
  std::string flavor_text;
};

struct ItemConfig {
  std::array<ItemDefinitionConfig, to_index(ItemId::count)> items;
};

ItemConfig MakeDefaultItemConfig();

}  // namespace arelto

#endif
