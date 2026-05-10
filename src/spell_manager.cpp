// src/spell_manager.cpp
#include "spell_manager.h"
#include <iostream>
#include "config/spell_config_yaml.h"

namespace arelto {

SpellManager::SpellManager() {}
SpellManager::~SpellManager() {}

void SpellManager::Initialize() {
  YAML::Node node;
  try {
    node = YAML::LoadFile("assets/config/spells.yaml");
  } catch (const YAML::Exception& e) {
    std::cerr << "Failed to load spells config: " << e.what() << "\n";
    return;
  }

  if (!node.IsSequence()) {
    std::cerr << "Spells config root must be a sequence\n";
    return;
  }

  spells_.clear();
  texture_ids_.clear();

  for (size_t i = 0; i < node.size(); ++i) {
    SpellConfig cfg;
    if (!YAML::convert<SpellConfig>::decode(node[i], cfg)) {
      std::cerr << "Failed to decode spell at index " << i << "\n";
      continue;
    }

    auto spell = std::make_unique<BaseProjectileSpell>(cfg.name);
    spell->SetId(static_cast<SpellId>(i));
    spell->SetCooldown(cfg.cooldown);
    spell->SetSpeed(cfg.speed);
    spell->SetDamage(cfg.damage);
    spell->SetWidth(cfg.width);
    spell->SetSpriteCellSize({cfg.sprite_cell_width, cfg.sprite_cell_height});
    spell->CaptureBaseStats();
    spells_.push_back(std::move(spell));
    texture_ids_.push_back(cfg.name);
  }
}

BaseProjectileSpell* SpellManager::GetSpell(SpellId id) {
  if (id < 0 || static_cast<size_t>(id) >= spells_.size()) {
    return nullptr;
  }
  return spells_[id].get();
}

const BaseProjectileSpell* SpellManager::GetSpell(SpellId id) const {
  if (id < 0 || static_cast<size_t>(id) >= spells_.size()) {
    return nullptr;
  }
  return spells_[id].get();
}

const std::vector<std::unique_ptr<BaseProjectileSpell>>&
SpellManager::GetAllSpells() const {
  return spells_;
}

size_t SpellManager::GetSpellCount() const {
  return spells_.size();
}

SpellTextureMapping SpellManager::GetSpellTextureMapping() const {
  SpellTextureMapping mapping;
  mapping.reserve(texture_ids_.size());
  for (size_t i = 0; i < texture_ids_.size(); ++i) {
    mapping.emplace_back(static_cast<SpellId>(i), texture_ids_[i]);
  }
  return mapping;
}

std::vector<std::string> SpellManager::GetSpellNames() const {
  std::vector<std::string> names;
  names.reserve(spells_.size());
  for (const auto& spell : spells_) {
    names.push_back(spell->GetName());
  }
  return names;
}

void SpellManager::ResetSpellStats() {
  for (const auto& spell : spells_) {
    spell->ResetStatsToBase();
  }
}

}  // namespace arelto
