#ifndef RL2_SPELL_MANAGER_H_
#define RL2_SPELL_MANAGER_H_

#include <memory>
#include <string>
#include <vector>
#include "abilities.h"

namespace arelto {

class SpellManager {
 public:
  using SpellList = std::vector<std::unique_ptr<BaseProjectileSpell>>;
  SpellManager();
  ~SpellManager();

  void Initialize();

  BaseProjectileSpell* GetSpell(SpellId id);
  const BaseProjectileSpell* GetSpell(SpellId id) const;
  const SpellList& GetAllSpells() const;
  size_t GetSpellCount() const;
  std::vector<std::string> GetSpellNames() const;
  SpellTextureMapping GetSpellTextureMapping() const;
  void ResetSpellStats();

 private:
  SpellList spells_;
  std::vector<std::string> texture_ids_;
};

}  // namespace arelto

#endif
