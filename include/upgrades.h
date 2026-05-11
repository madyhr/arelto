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
#include "utils.h"

namespace arelto {

// One row on an upgrade card: a textual description plus an
// optional numeric before/after pair. Used for UI rendering.
// NOTE: For rows with empty old_value / new_value only the description is rendered.
struct UpgradeDisplayRow {
  std::string description;
  std::string old_value;
  std::string new_value;
  bool is_improvement = true;
};

class Upgrade {
 public:
  virtual ~Upgrade() = default;

  virtual void Apply(Player& player) = 0;
  virtual std::string GetName() const = 0;
  virtual std::vector<UpgradeDisplayRow> GetDisplayRows() const = 0;
};

using UpgradeOptions = std::vector<std::unique_ptr<Upgrade>>;

namespace {
inline bool IsHigherBetter(SpellUpgradeType type) {
  switch (type) {
    case SpellUpgradeType::size:
    case SpellUpgradeType::cooldown:
    case SpellUpgradeType::damage:
    case SpellUpgradeType::speed:
    default:
      return true;
  }
}
}  // namespace

const Stat* ResolveSpellStat(const BaseProjectileSpell& player,
                             SpellUpgradeType stat_type);
Stat* ResolveSpellStat(BaseProjectileSpell& player, SpellUpgradeType stat_type);

std::string ResolveSpellUpgradeDescription(SpellUpgradeType stat_type);

struct SpellStatSpec {
  SpellUpgradeType stat_type;
  ModifierType modifier_type;
  float value;
  std::string description;
};

struct SpellStatModifier {
  SpellUpgradeType stat_type;
  ModifierType modifier_type;
  float raw_value;
  ValueRange value_range;
  std::string description;
  bool is_higher_better;
};

struct SpellUpgrade {
  SpellId id;
  Rarity rarity;
  std::vector<SpellStatSpec> stat_specs;
};

class SpellStatUpgrade : public Upgrade {
 public:
  SpellStatUpgrade(SpellId spell_id, std::string spell_name,
                   std::vector<SpellStatModifier> stat_modifiers,
                   Size2D sprite_cell_size = {})
      : spell_id_(spell_id),
        spell_name_(ToTitleCase(std::move(spell_name))),
        stat_modifiers_(std::move(stat_modifiers)),
        sprite_cell_size_(sprite_cell_size) {}

  void Apply(Player& player) override {
    BaseProjectileSpell* spell = player.GetSpell(spell_id_);
    if (!spell) {
      return;
    }

    for (const SpellStatModifier& stat_modifier : stat_modifiers_) {
      Stat* stat_to_upgrade = ResolveSpellStat(*spell, stat_modifier.stat_type);
      if (stat_to_upgrade == nullptr) {
        continue;
      }
      Modifier modifier{stat_modifier.raw_value, stat_modifier.modifier_type,
                        nullptr};
      stat_to_upgrade->AddModifier(modifier);
    }
    player.spell_stats_.SetProjectileSpellStats(*spell);
  }

  SpellUpgradeType GetType() const { return type_; }
  SpellId GetSpellID() const { return spell_id_; }
  Size2D GetSpriteCellSize() const { return sprite_cell_size_; }

  std::string GetName() const override { return spell_name_; }

  std::vector<UpgradeDisplayRow> GetDisplayRows() const override {
    std::vector<UpgradeDisplayRow> rows;
    rows.reserve(stat_modifiers_.size());
    for (const SpellStatModifier& stat_modifier : stat_modifiers_) {
      bool value_increased =
          stat_modifier.value_range.updated > stat_modifier.value_range.current;
      bool is_improvement =
          (value_increased && stat_modifier.is_higher_better) ||
          (!value_increased && !stat_modifier.is_higher_better);
      rows.push_back(UpgradeDisplayRow{
          stat_modifier.description,
          FormatValue(stat_modifier.value_range.current),
          FormatValue(stat_modifier.value_range.updated), is_improvement});
    }
    return rows;
  }

 private:
  static std::string FormatValue(float value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << value;
    return ss.str();
  }

  SpellId spell_id_;
  std::string spell_name_;
  SpellUpgradeType type_ = SpellUpgradeType::damage;
  std::vector<SpellStatModifier> stat_modifiers_;
  float current_value_;
  float updated_value_;
  Size2D sprite_cell_size_;
};

}  // namespace arelto

#endif
