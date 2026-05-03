#ifndef RL2_CONFIG_SPELL_CONFIG_H_
#define RL2_CONFIG_SPELL_CONFIG_H_

#include <cstdint>
#include <string>
#include "types.h"

namespace arelto {

struct SpellConfig {
  std::string name;
  uint32_t width = 60;
  uint32_t height = 60;
  uint32_t sprite_cell_width = 60;
  uint32_t sprite_cell_height = 60;
  float speed = 500.0f;
  int damage = 5;
  float cooldown = 1.0f;
  Collider collider;
};

}  // namespace arelto

#endif
