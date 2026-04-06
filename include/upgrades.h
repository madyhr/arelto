#ifndef RL2_UPGRADES_H_
#define RL2_UPGRADES_H_

#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "abilities.h"
#include "entity.h"
#include "types.h"

namespace arelto {

// One row on an upgrade card: a textual description plus an
// optional numeric before/after pair. Used for UI rendering.
// NOTE: For rows with empty old_value / new_value only the description is rendered.
struct UpgradeDisplayRow {
  std::string description;
  std::string old_value;
  std::string new_value;
};

class Upgrade {
 public:
  virtual ~Upgrade() = default;

  virtual void Apply(Player& player) = 0;
  virtual std::string GetName() const = 0;
  virtual std::vector<UpgradeDisplayRow> GetDisplayRows() const = 0;
};

using UpgradeOptions = std::vector<std::unique_ptr<Upgrade>>;

class SpellStatUpgrade : public Upgrade {
 public:
  SpellStatUpgrade(SpellId spell_id, std::string spell_name,
                   SpellUpgradeType type, ValueRange value_range)
      : spell_id_(spell_id),
        spell_name_(std::move(spell_name)),
        type_(type),
        current_value_(value_range.current),
        updated_value_(value_range.updated) {}

  void Apply(Player& player) override {
    BaseProjectileSpell* spell = player.GetSpell(spell_id_);
    if (!spell) {
      return;
    }

    spell->ModifyStat(type_, updated_value_);

    player.spell_stats_.SetProjectileSpellStats(*spell);
  }

  SpellUpgradeType GetType() const { return type_; }
  SpellId GetSpellID() const { return spell_id_; }

  std::string GetName() const override { return spell_name_; }

  std::vector<UpgradeDisplayRow> GetDisplayRows() const override {
    return {UpgradeDisplayRow{GetDescriptionForType(type_),
                              FormatValue(current_value_),
                              FormatValue(updated_value_)}};
  }

 private:
  static std::string GetDescriptionForType(SpellUpgradeType type) {
    switch (type) {
      case SpellUpgradeType::damage:
        return "Increase Damage";
      case SpellUpgradeType::speed:
        return "Increase Speed";
      case SpellUpgradeType::cooldown:
        return "Decrease Cooldown";
      case SpellUpgradeType::size:
        return "Increase Size";
      case SpellUpgradeType::count:
        return "Unknown Upgrade";
    }
    return "Unknown Upgrade";
  }

  static std::string FormatValue(float value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << value;
    return ss.str();
  }

  SpellId spell_id_;
  std::string spell_name_;
  SpellUpgradeType type_;
  float current_value_;
  float updated_value_;
};

}  // namespace arelto

#endif
