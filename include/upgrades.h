#ifndef RL2_UPGRADES_H_
#define RL2_UPGRADES_H_

#include <iomanip>
#include <sstream>
#include <string>
#include "abilities.h"
#include "entity.h"
#include "types.h"

namespace rl2 {

class Upgrade {
 public:
  virtual ~Upgrade() = default;

  virtual void Apply(Player& player) = 0;
  virtual std::string GetDescription() const = 0;
  virtual UpgradeType GetType() const = 0;
  virtual SpellId GetSpellID() const = 0;
  virtual std::string GetSpellName() const = 0;
  virtual std::string GetOldValueString() const = 0;
  virtual std::string GetNewValueString() const = 0;
};

class SpellStatUpgrade : public Upgrade {
 public:
  SpellStatUpgrade(SpellId spell_id, std::string spell_name, UpgradeType type,
                   float current_value, float new_value)
      : spell_id_(spell_id),
        spell_name_(spell_name),
        type_(type),
        current_value_(current_value),
        new_value_(new_value) {}

  void Apply(Player& player) override {
    BaseProjectileSpell* spell = player.GetSpell(spell_id_);
    if (!spell) {
      return;
    }

    spell->ModifyStat(type_, new_value_);

    player.spell_stats_.SetProjectileSpellStats(*spell);
  }

  std::string GetDescription() const override {
    switch (type_) {
      case UpgradeType::damage:
        return "Increase Damage";
      case UpgradeType::speed:
        return "Increase Speed";
      case UpgradeType::cooldown:
        return "Decrease Cooldown";
      case UpgradeType::size:
        return "Increase Size";
      default:
        return "Unknown Upgrade";
    }
  }

  UpgradeType GetType() const override { return type_; }
  SpellId GetSpellID() const override { return spell_id_; }

  std::string GetSpellName() const override { return spell_name_; }

  std::string GetOldValueString() const override {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << current_value_;
    return ss.str();
  }

  std::string GetNewValueString() const override {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << new_value_;
    return ss.str();
  }

 private:
  SpellId spell_id_;
  std::string spell_name_;
  UpgradeType type_;
  float current_value_;
  float new_value_;
};

}  // namespace rl2

#endif
