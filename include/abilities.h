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
  Stat speed_;
  float inv_mass_ = 0.0f;
  Stat damage_;
  StatsSize size_;
  Size2D sprite_cell_size_;
  std::string name_;
  float base_speed_ = 0.0f;
  float base_inv_mass_ = 0.0f;
  float base_damage_ = 0.0f;
  float base_cooldown_ = 1.0f;
  float base_width_ = 0.0f;

 public:
  BaseProjectileSpell() = default;
  explicit BaseProjectileSpell(std::string name) : name_(std::move(name)) {};
  float GetSpeed() { return speed_.GetValue(); };
  void SetSpeed(float speed) { speed_.SetBaseValue(speed); };
  float GetInvMass() { return inv_mass_; };
  void SetInvMass(float inv_mass) { inv_mass_ = inv_mass; };
  float GetDamage() { return damage_.GetValue(); };
  void SetDamage(float damage) { damage_.SetBaseValue(damage); };
  Collider GetCollider() { return size_.GetCollider(); };
  Size2D GetSize() const { return size_.GetSize(); };
  void SetWidth(float width) { size_.SetBaseWidth(width); };
  Size2D GetSpriteCellSize() const { return sprite_cell_size_; };
  void SetSpriteCellSize(Size2D size) { sprite_cell_size_ = size; };
  std::string GetName() const { return name_; };

  void CaptureBaseStats() {
    base_speed_ = speed_.GetValue();
    base_inv_mass_ = inv_mass_;
    base_damage_ = damage_.GetValue();
    base_cooldown_ = GetCooldown();
    base_width_ = size_.width_.GetValue();
  }

  void ResetStatsToBase() {
    SetSpeed(base_speed_);
    SetInvMass(base_inv_mass_);
    SetDamage(base_damage_);
    SetCooldown(base_cooldown_);
    SetWidth(base_width_);
    SetTimeOfLastUse(-1.0f);
  }

  virtual void ModifyStat(SpellUpgradeType type, float value) {
    switch (type) {
      case SpellUpgradeType::damage:
        SetDamage(value);
        break;
      case SpellUpgradeType::speed:
        SetSpeed(value);
        break;
      case SpellUpgradeType::cooldown:
        SetCooldown(value);
        break;
      case SpellUpgradeType::size: {
        float current_w = static_cast<float>(GetSize().width);
        float current_h = static_cast<float>(GetSize().height);
        if (current_w > 0) {
          float ratio = current_h / current_w;
          float new_w = value;
          float new_h = new_w * ratio;
          SetWidth(new_w);
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
  std::vector<float> damage;

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
    sprite_size[id] = spell.GetSize();
    damage[id] = spell.GetDamage();
  };

  void ResetProjectileSpellStats(BaseProjectileSpell& spell) {
    SpellId id = spell.GetId();
    time_of_last_use[id] = 0.0f;
  }
};

}  // namespace arelto
#endif
