#ifndef RL2_UPGRADES_H_
#define RL2_UPGRADES_H_

#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include "abilities.h"
#include "entity.h"
#include "types.h"

namespace arelto {

class Upgrade {
 public:
  virtual ~Upgrade() = default;

  virtual void Apply(Player& player) = 0;
  virtual std::string GetDescription() const = 0;
  virtual std::string GetName() const = 0;
  virtual std::string GetOldValueString() const = 0;
  virtual std::string GetNewValueString() const = 0;
};

using UpgradeOptions = std::vector<std::unique_ptr<Upgrade>>;

class SpellStatUpgrade : public Upgrade {
 public:
  SpellStatUpgrade(SpellId spell_id, std::string spell_name,
                   SpellUpgradeType type, float current_value, float new_value)
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
      case SpellUpgradeType::damage:
        return "Increase Damage";
      case SpellUpgradeType::speed:
        return "Increase Speed";
      case SpellUpgradeType::cooldown:
        return "Decrease Cooldown";
      case SpellUpgradeType::size:
        return "Increase Size";
      default:
        return "Unknown Upgrade";
    }
  }

  SpellUpgradeType GetType() const { return type_; }
  SpellId GetSpellID() const { return spell_id_; }

  std::string GetName() const override { return spell_name_; }

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
  SpellUpgradeType type_;
  float current_value_;
  float new_value_;
};

}  // namespace arelto

#endif
