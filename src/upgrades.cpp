// src/upgrades.cpp
#include "upgrades.h"
#include "abilities.h"
#include "types.h"

namespace arelto {

const Stat* ResolveSpellStat(const BaseProjectileSpell& spell,
                             SpellUpgradeType stat_type) {
  switch (stat_type) {
    case SpellUpgradeType::damage:
      return &spell.damage_;
    case SpellUpgradeType::size:
      return &spell.size_.width_;
    case SpellUpgradeType::speed:
      return &spell.speed_;
    case SpellUpgradeType::cooldown:
      return &spell.cooldown_;
    case SpellUpgradeType::count:
      return nullptr;
  }
  return nullptr;
}

Stat* ResolveSpellStat(BaseProjectileSpell& spell, SpellUpgradeType stat_type) {
  return const_cast<Stat*>(ResolveSpellStat(
      static_cast<const BaseProjectileSpell&>(spell), stat_type));
}

std::string ResolveSpellUpgradeDescription(SpellUpgradeType stat_type) {
  switch (stat_type) {
    case SpellUpgradeType::damage:
      return "Increase Spell Damage";
    case SpellUpgradeType::size:
      return "Increase Spell Size";
    case SpellUpgradeType::speed:
      return "Increase Projectile Speed";
    case SpellUpgradeType::cooldown:
      return "Decrease Spell Cooldown";
    case SpellUpgradeType::count:
      return "";
  }
  return "";
}

ModifierType ResolveSpellUpgradeModifierType(SpellUpgradeType stat_type) {
  switch (stat_type) {
    case SpellUpgradeType::damage:
    case SpellUpgradeType::size:
    case SpellUpgradeType::speed:
    case SpellUpgradeType::cooldown:
      return ModifierType::percent_mult;
    case SpellUpgradeType::count:
      return ModifierType::flat;
  }
  return ModifierType::flat;
}

float ResolveSpellUpgradeModifierValue(SpellUpgradeType stat_type) {
  switch (stat_type) {
    case SpellUpgradeType::damage:
    case SpellUpgradeType::size:
    case SpellUpgradeType::speed:
      return 0.05f;
    case SpellUpgradeType::cooldown:
      return -0.05f;
    case SpellUpgradeType::count:
      return 0.0f;
  }
  return 0.0f;
}

}  // namespace arelto
