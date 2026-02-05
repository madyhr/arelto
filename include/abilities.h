// include/abilities.h
#ifndef RL2_ABILITIES_H_
#define RL2_ABILITIES_H_
#include <SDL2/SDL_render.h>
#include <array>
#include <cstdint>
#include <string>
#include "constants/projectile.h"
#include "types.h"

namespace arelto {

enum SpellId : int { FireballId, FrostboltId };

class BaseSpell {
  SpellId id_;

 public:
  float cooldown_ = 1.0f;
  float time_of_last_use_ = -1.0f;
  explicit BaseSpell(SpellId id) : id_(id) {};
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
  Size2D sprite_size_;
  std::string name_;

 public:
  BaseProjectileSpell(SpellId id, std::string name)
      : BaseSpell(id), name_(name) {};
  virtual float GetSpeed() { return speed_; };
  virtual void SetSpeed(float speed) { speed_ = speed; };
  virtual float GetInvMass() { return inv_mass_; };
  virtual void SetInvMass(float inv_mass) { inv_mass_ = inv_mass; };
  virtual uint32_t GetDamage() { return damage_; };
  virtual void SetDamage(uint32_t damage) { damage_ = damage; };
  virtual Collider GetCollider() { return collider_; };
  virtual void SetCollider(Collider collider) { collider_ = collider; };
  virtual Size2D GetSpriteSize() { return sprite_size_; };
  virtual void SetSpriteSize(Size2D size) { sprite_size_ = size; };
  std::string GetName() const { return name_; };

  virtual void ModifyStat(UpgradeType type, float value) {
    switch (type) {
      case UpgradeType::damage:
        SetDamage(static_cast<uint32_t>(value));
        break;
      case UpgradeType::speed:
        SetSpeed(value);
        break;
      case UpgradeType::cooldown:
        SetCooldown(value);
        break;
      case UpgradeType::size: {
        float current_w = static_cast<float>(GetSpriteSize().width);
        float current_h = static_cast<float>(GetSpriteSize().height);
        if (current_w > 0) {
          float ratio = current_h / current_w;
          float new_w = value;
          float new_h = new_w * ratio;
          SetSpriteSize(
              {static_cast<uint32_t>(new_w), static_cast<uint32_t>(new_h)});
          // TODO:This is a placeholder way to set the collider. May require
          // reworking how Colliders are defined in the worst case.
          SetCollider({{static_cast<float>(0.5f * new_w),
                        static_cast<float>(0.5f * new_h)},
                       {static_cast<uint32_t>(new_w * 0.8f),
                        static_cast<uint32_t>(new_h * 0.8f)}});
        }
        break;
      }
    }
  }
};

class Fireball : public BaseProjectileSpell {
 public:
  Fireball() : BaseProjectileSpell(SpellId::FireballId, "Fireball") {
    SetCooldown(kFireballCooldown);
    SetSpeed(kFireballSpeed);
    SetCollider({{0.5 * kFireballSpriteWidth, 0.5 * kFireballSpriteHeight},
                 {kFireballColliderWidth, kFireballColliderHeight}});
    SetSpriteSize({kFireballSpriteWidth, kFireballSpriteHeight});
    SetDamage(kFireBallDamage);
  }
};

class Frostbolt : public BaseProjectileSpell {
 public:
  Frostbolt() : BaseProjectileSpell(SpellId::FrostboltId, "Frostbolt") {
    SetCooldown(kFrostboltCooldown);
    SetSpeed(kFrostboltSpeed);
    SetCollider({{0.5 * kFrostboltSpriteWidth, 0.5 * kFrostboltSpriteHeight},
                 {kFrostboltColliderWidth, kFrostboltColliderHeight}});
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
  std::array<Size2D, N> sprite_size;
  std::array<int, N> damage;

  void SetProjectileSpellStats(BaseProjectileSpell& spell) {
    SpellId id = spell.GetId();
    cooldown[id] = spell.GetCooldown();
    speed[id] = spell.GetSpeed();
    collider[id] = spell.GetCollider();
    sprite_size[id] = spell.GetSpriteSize();
    damage[id] = spell.GetDamage();
  };

  void ResetProjectileSpellStats(BaseProjectileSpell& spell) {
    SpellId id = spell.GetId();
    time_of_last_use[id] = 0.0f;
  }
};
};  // namespace arelto
#endif
