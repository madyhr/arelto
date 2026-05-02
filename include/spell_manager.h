#ifndef RL2_SPELL_MANAGER_H_
#define RL2_SPELL_MANAGER_H_

#include <memory>
#include <string>
#include <vector>
#include "abilities.h"

namespace arelto {

class SpellManager {
 public:
  SpellManager();
  ~SpellManager();

  void Initialize();

  BaseProjectileSpell* GetSpell(SpellId id);
  const BaseProjectileSpell* GetSpell(SpellId id) const;
  const std::vector<std::unique_ptr<BaseProjectileSpell>>& GetAllSpells() const;
  size_t GetSpellCount() const;
  std::vector<std::string> GetTexturePaths() const;
  std::vector<std::string> GetSpellNames() const;

 private:
  std::vector<std::unique_ptr<BaseProjectileSpell>> spells_;
  std::vector<std::string> texture_paths_;
};

}  // namespace arelto

#endif
