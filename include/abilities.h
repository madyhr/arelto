// include/types.h
#ifndef RL2_ABILITIES_H_
#define RL2_ABILITIES_H_
#include <SDL2/SDL_render.h>
#include <constants.h>

namespace rl2 {

struct Spell {
  float cooldown;
  float time_of_last_use;
  int spell_id;

  float GetReadyTime() const { return cooldown + time_of_last_use; };
};

struct Fireball : Spell {
  int spell_id = 0;
};

struct Frostbolt : Spell {
  int spell_id = 0;
};

}  // namespace rl2
#endif
