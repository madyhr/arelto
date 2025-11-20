// include/types.h
#ifndef RL2_ABILITIES_H_
#define RL2_ABILITIES_H_
#include <SDL2/SDL_render.h>
#include <constants.h>

namespace rl2 {

struct BaseSpell {
  float cooldown = 1.0f;
  float time_of_last_use = 0.0f;
  int id = -1;

  float GetReadyTime() const { return cooldown + time_of_last_use; };
};

struct BaseProjectileSpell : BaseSpell {
  float base_speed = 100.0f;
  uint32_t base_width = 0;
  uint32_t base_height = 0;
};

struct Fireball : BaseProjectileSpell {
  Fireball() {
    id = 0;
    base_speed = kFireballSpeed;
    base_width = kFireballWidth;
    base_height = kFireballHeight;
  }
};

struct Frostbolt : BaseProjectileSpell {
  Frostbolt() {
    id = 1;
    base_speed = kFrostboltSpeed;
    base_width = kFrostboltWidth;
    base_height = kFrostboltHeight;
  }
};

}  // namespace rl2
#endif
