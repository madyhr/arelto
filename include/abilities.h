// include/types.h
#ifndef RL2_ABILITIES_H_
#define RL2_ABILITIES_H_
#include <SDL2/SDL_render.h>
#include <constants.h>
#include <cstdint>

namespace rl2 {

struct Spell {
  float cooldown;
  float time_of_last_use;

  float GetCDTime() const { return cooldown + time_of_last_use; };
};

struct Fireball : Spell {
};

}  // namespace rl2
#endif
