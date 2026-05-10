#ifndef RL2_CONFIG_SPELL_CONFIG_H_
#define RL2_CONFIG_SPELL_CONFIG_H_

#include <cstdint>
#include <string>

namespace arelto {

struct SpellConfig {
  std::string name;
  float width = 60.0f;
  float aspect_ratio = 1.0f;
  uint32_t sprite_cell_width = 60;
  uint32_t sprite_cell_height = 60;
  float speed = 500.0f;
  float damage = 5.0f;
  float cooldown = 1.0f;
};

}  // namespace arelto

#endif
