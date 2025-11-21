// include/types.h
#ifndef RL2_ABILITIES_H_
#define RL2_ABILITIES_H_
#include <SDL2/SDL_render.h>
#include <constants.h>

namespace rl2 {

struct BaseSpell {
  float cooldown = 1.0f;
  float time_of_last_use = -1.0f;
  int id = -1;

  float GetReadyTime() const { return cooldown + time_of_last_use; };
};

struct BaseProjectileSpell : BaseSpell {
  float speed = 0.0f;
  float inv_mass = 0.0f;
  uint32_t width = 0;
  uint32_t height = 0;
};

struct Fireball : BaseProjectileSpell {
  Fireball() {
    id = 0;
    speed = kFireballSpeed;
    width = kFireballWidth;
    height = kFireballHeight;
  }
};

struct Frostbolt : BaseProjectileSpell {
  Frostbolt() {
    id = 1;
    speed = kFrostboltSpeed;
    width = kFrostboltWidth;
    height = kFrostboltHeight;
  }
};

}  // namespace rl2
#endif
