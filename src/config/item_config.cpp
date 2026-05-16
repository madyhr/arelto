#include "config/item_config.h"

namespace arelto {

ItemConfig MakeDefaultItemConfig() {
  ItemConfig config{};
  config.items[to_index(ItemId::elia_armor_plate)].flavor_text =
      ItemFlavorText.at(ItemId::elia_armor_plate);
  config.items[to_index(ItemId::damodei_claw)].flavor_text =
      ItemFlavorText.at(ItemId::damodei_claw);
  config.items[to_index(ItemId::volmnih_boots)].flavor_text =
      ItemFlavorText.at(ItemId::volmnih_boots);
  config.items[to_index(ItemId::sarto_button_bible)].flavor_text =
      ItemFlavorText.at(ItemId::sarto_button_bible);
  config.items[to_index(ItemId::aiayn_scale)].flavor_text =
      ItemFlavorText.at(ItemId::aiayn_scale);
  return config;
}

}  // namespace arelto
