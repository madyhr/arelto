// include/abilities.h
#ifndef RL2_ABILITIES_H_
#define RL2_ABILITIES_H_
#include <SDL2/SDL_render.h>
#include <array>
#include <cstdint>
#include "constants.h"
#include "types.h"

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
  uint32_t damage_ = 0;
  Collider collider_;
  Size sprite_size_;

 public:
  BaseProjectileSpell(SpellId id) : BaseSpell(id){};
  virtual float GetSpeed() { return speed_; };
  virtual void SetSpeed(float speed) { speed_ = speed; };
  virtual float GetInvMass() { return inv_mass_; };
  virtual void SetInvMass(float inv_mass) { inv_mass_ = inv_mass; };
  virtual uint32_t GetDamage() { return damage_; };
  virtual void SetDamage(uint32_t damage) { damage_ = damage; };
  virtual Collider GetCollider() { return collider_; };
  virtual void SetCollider(Collider collider) { collider_ = collider; };
  virtual Size GetSpriteSize() { return sprite_size_; };
  virtual void SetSpriteSize(Size size) { sprite_size_ = size; };
};

class Fireball : public BaseProjectileSpell {
 public:
  Fireball() : BaseProjectileSpell(SpellId::FireballId) {
    SetCooldown(kFireballCooldown);
    SetSpeed(kFireballSpeed);
    SetCollider({{0.5f * kFireballSpriteWidth, 0.5f * kFireballSpriteHeight},
                 {kFireballSpriteWidth, kFireballSpriteHeight}});
    SetSpriteSize({kFireballSpriteWidth, kFireballSpriteHeight});
    SetDamage(kFireBallDamage);
  }
};

class Frostbolt : public BaseProjectileSpell {
 public:
  Frostbolt() : BaseProjectileSpell(SpellId::FrostboltId) {
    SetCooldown(kFrostboltCooldown);
    SetSpeed(kFrostboltSpeed);
    SetCollider({{0.5f * kFrostboltSpriteWidth, 0.5f * kFrostboltSpriteHeight},
                 {kFrostboltColliderWidth, kFrostboltColliderHeight }});
    SetSpriteSize({kFrostboltSpriteWidth, kFrostboltSpriteHeight});
    SetDamage(kFrostboltDamage);
  }
};

template <size_t N>
struct SpellStats {
  std::array<float, N> cooldown;
  std::array<float, N> time_of_last_use;
  std::array<float, N> speed;
  std::array<Collider, N> collider;
  std::array<Size, N> sprite_size;
  std::array<int, N> damage;

  void SetProjectileSpellStats(BaseProjectileSpell spell) {
    int id = spell.GetId();
    cooldown[id] = spell.GetCooldown();
    speed[id] = spell.GetSpeed();
    collider[id] = spell.GetCollider();
    sprite_size[id] = spell.GetSpriteSize();
    damage[id] = spell.GetDamage();
  };
};
};  // namespace rl2
#endif
