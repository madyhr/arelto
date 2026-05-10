// include/abilities.h
#ifndef RL2_ABILITIES_H_
#define RL2_ABILITIES_H_
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "types.h"

namespace arelto {

using SpellId = int;
using SpellTextureMapping = std::vector<std::pair<SpellId, std::string>>;

class BaseSpell {
 private:
  SpellId id_ = -1;
  float cooldown_ = 1.0f;
  float time_of_last_use_ = -1.0f;

 public:
  BaseSpell() = default;
  virtual ~BaseSpell() = default;
  SpellId GetId() { return id_; };
  void SetId(SpellId id) { id_ = id; };
  float GetCooldown() { return cooldown_; };
  void SetCooldown(float cooldown) { cooldown_ = cooldown; };
  void SetTimeOfLastUse(float time) { time_of_last_use_ = time; };
  float GetTimeOfLastUse() { return time_of_last_use_; };
};

class BaseProjectileSpell : public BaseSpell {
 private:
  float speed_ = 0.0f;
  float inv_mass_ = 0.0f;
  int damage_ = 0;
  Collider collider_;
  Size2D sprite_size_;
  Size2D sprite_cell_size_;
  std::string name_;
  float base_speed_ = 0.0f;
  float base_inv_mass_ = 0.0f;
  int base_damage_ = 0;
  float base_cooldown_ = 1.0f;
  Collider base_collider_;
  Size2D base_sprite_size_;

 public:
  BaseProjectileSpell() = default;
  explicit BaseProjectileSpell(std::string name) : name_(std::move(name)) {};
  float GetSpeed() { return speed_; };
  void SetSpeed(float speed) { speed_ = speed; };
  float GetInvMass() { return inv_mass_; };
  void SetInvMass(float inv_mass) { inv_mass_ = inv_mass; };
  int GetDamage() { return damage_; };
  void SetDamage(int damage) { damage_ = damage; };
  Collider GetCollider() { return collider_; };
  void SetCollider(Collider collider) { collider_ = collider; };
  Size2D GetSpriteSize() const { return sprite_size_; };
  void SetSpriteSize(Size2D size) { sprite_size_ = size; };
  Size2D GetSpriteCellSize() const { return sprite_cell_size_; };
  void SetSpriteCellSize(Size2D size) { sprite_cell_size_ = size; };
  std::string GetName() const { return name_; };

  void CaptureBaseStats() {
    base_speed_ = speed_;
    base_inv_mass_ = inv_mass_;
    base_damage_ = damage_;
    base_cooldown_ = GetCooldown();
    base_collider_ = collider_;
    base_sprite_size_ = sprite_size_;
  }

  void ResetStatsToBase() {
    SetSpeed(base_speed_);
    SetInvMass(base_inv_mass_);
    SetDamage(base_damage_);
    SetCooldown(base_cooldown_);
    SetCollider(base_collider_);
    SetSpriteSize(base_sprite_size_);
    SetTimeOfLastUse(-1.0f);
  }

  virtual void ModifyStat(SpellUpgradeType type, float value) {
    switch (type) {
      case SpellUpgradeType::damage:
        SetDamage(static_cast<int>(value));
        break;
      case SpellUpgradeType::speed:
        SetSpeed(value);
        break;
      case SpellUpgradeType::cooldown:
        SetCooldown(value);
        break;
      case SpellUpgradeType::size: {
        float current_w = static_cast<float>(GetSpriteSize().width);
        float current_h = static_cast<float>(GetSpriteSize().height);
        if (current_w > 0) {
          float ratio = current_h / current_w;
          float new_w = value;
          float new_h = new_w * ratio;
          SetSpriteSize(
              {static_cast<uint32_t>(new_w), static_cast<uint32_t>(new_h)});
          SetCollider({{static_cast<float>(0.5f * new_w),
                        static_cast<float>(0.5f * new_h)},
                       {static_cast<uint32_t>(new_w * 0.8f),
                        static_cast<uint32_t>(new_h * 0.8f)}});
        }
        break;
      }
      case SpellUpgradeType::count:
        break;
    }
  }
};

struct SpellStats {
  std::vector<float> cooldown;
  std::vector<float> time_of_last_use;
  std::vector<float> speed;
  std::vector<Collider> collider;
  std::vector<Size2D> sprite_size;
  std::vector<int> damage;

  void Resize(size_t n) {
    cooldown.resize(n);
    time_of_last_use.resize(n);
    speed.resize(n);
    collider.resize(n);
    sprite_size.resize(n);
    damage.resize(n);
  }

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

}  // namespace arelto
#endif
