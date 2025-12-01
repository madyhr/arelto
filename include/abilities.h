// include/abilities.h
#ifndef RL2_ABILITIES_H_
#define RL2_ABILITIES_H_
#include <SDL2/SDL_render.h>
#include <array>
#include <cstdint>
#include "constants.h"

namespace rl2 {

enum SpellId : int { FireballId, FrostboltId };

class BaseSpell {
  SpellId id_;

 public:
  float cooldown_ = 1.0f;
  float time_of_last_use_ = -1.0f;
  explicit BaseSpell(SpellId id) : id_(id){};
  virtual ~BaseSpell() = default;
  virtual SpellId GetId() { return id_; };
  virtual float GetCooldown() { return cooldown_; };
  virtual void SetCooldown(float cooldown) { cooldown_ = cooldown; };
  virtual void SetTimeOfLastUse(float time) { time_of_last_use_ = time; };
  virtual float GetReadyTime() const { return cooldown_ + time_of_last_use_; };
};

class BaseProjectileSpell : public BaseSpell {
  float speed_ = 0.0f;
  float inv_mass_ = 0.0f;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t damage_ = 0;

 public:
  BaseProjectileSpell(SpellId id) : BaseSpell(id){};
  virtual float GetSpeed() { return speed_; };
  virtual void SetSpeed(float speed) { speed_ = speed; };
  virtual float GetInvMass() { return inv_mass_; };
  virtual void SetInvMass(float inv_mass) { inv_mass_ = inv_mass; };
  virtual uint32_t GetWidth() { return width_; };
  virtual void SetWidth(uint32_t width) { width_ = width; };
  virtual uint32_t GetHeight() { return height_; };
  virtual void SetHeight(uint32_t height) { height_ = height; };
  virtual uint32_t GetDamage() { return damage_; };
  virtual void SetDamage(uint32_t damage) { damage_ = damage; };
};

class Fireball : public BaseProjectileSpell {
 public:
  Fireball() : BaseProjectileSpell(SpellId::FireballId) {
    SetCooldown(kFireballCooldown);
    SetSpeed(kFireballSpeed);
    SetWidth(kFireballWidth);
    SetHeight(kFireballHeight);
    SetDamage(kFireBallDamage);
  }
};

class Frostbolt : public BaseProjectileSpell {
 public:
  Frostbolt() : BaseProjectileSpell(SpellId::FrostboltId) {
    SetCooldown(kFrostboltCooldown);
    SetSpeed(kFrostboltSpeed);
    SetWidth(kFrostboltWidth);
    SetHeight(kFrostboltHeight);
    SetDamage(kFrostboltDamage);
  }
};

template <size_t N>
struct SpellStats {
  std::array<float, N> cooldown;
  std::array<float, N> time_of_last_use;
  std::array<float, N> speed;
  std::array<uint32_t, N> width;
  std::array<uint32_t, N> height;
  std::array<int, N> damage;

  void SetProjectileSpellStats(BaseProjectileSpell spell) {
    int id = spell.GetId();
    cooldown[id] = spell.GetCooldown();
    speed[id] = spell.GetSpeed();
    width[id] = spell.GetWidth();
    height[id] = spell.GetHeight();
    damage[id] = spell.GetDamage();
  };
};

};  // namespace rl2
#endif
