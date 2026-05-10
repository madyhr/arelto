// Regression tests for full run state reset.

#include <gtest/gtest.h>

#include "config/entity_config.h"
#include "items.h"
#include "scene.h"
#include "spell_manager.h"
#include "upgrades.h"

namespace arelto {
namespace {

Scene CreateSceneWithLocalSpellManager(SpellManager& spell_manager) {
  spell_manager.Initialize();

  Scene scene;
  scene.player.SetSpellManager(&spell_manager);
  scene.player.spell_stats_.Resize(spell_manager.GetSpellCount());
  scene.Reset(MakeDefaultEntityConfig());
  return scene;
}

TEST(SceneResetTest, ResetRestoresSpellUpgradeStats) {
  SpellManager spell_manager;
  Scene scene = CreateSceneWithLocalSpellManager(spell_manager);

  BaseProjectileSpell* spell = scene.player.GetSpell(0);
  ASSERT_NE(spell, nullptr);

  const int base_damage = spell->GetDamage();
  SpellStatUpgrade upgrade(0, spell->GetName(), SpellUpgradeType::damage,
                           ValueRange{static_cast<float>(base_damage),
                                      static_cast<float>(base_damage + 10)});
  upgrade.Apply(scene.player);
  ASSERT_EQ(scene.player.spell_stats_.damage[0], base_damage + 10);

  scene.Reset(MakeDefaultEntityConfig());

  EXPECT_EQ(spell->GetDamage(), base_damage);
  EXPECT_EQ(scene.player.spell_stats_.damage[0], base_damage);
}

TEST(SceneResetTest, ResetClearsItemStatModifiersInventoryAndOptions) {
  SpellManager spell_manager;
  Scene scene = CreateSceneWithLocalSpellManager(spell_manager);

  const EntityConfig config = MakeDefaultEntityConfig();
  const float base_movement_speed =
      scene.player.stats_.movement_speed.GetValue();

  scene.player.stats_.movement_speed.AddModifier(
      Modifier{0.5f, ModifierType::percent_mult, nullptr});
  scene.player.stats_.armor.SetBaseValue(3.0f);
  scene.player.stats_.armor.AddModifier(
      Modifier{2.0f, ModifierType::flat, nullptr});
  scene.player.AddToInventory(ItemId::volmnih_boots);
  scene.level_up_options.push_back(nullptr);
  scene.item_options.push_back(nullptr);

  ASSERT_GT(scene.player.stats_.movement_speed.GetValue(),
            base_movement_speed);
  ASSERT_FLOAT_EQ(scene.player.stats_.armor.GetValue(), 5.0f);
  ASSERT_FALSE(scene.player.inventory_.empty());
  ASSERT_FALSE(scene.level_up_options.empty());
  ASSERT_FALSE(scene.item_options.empty());

  scene.Reset(config);

  EXPECT_FLOAT_EQ(scene.player.stats_.movement_speed.GetValue(),
                  config.player.movement_speed);
  EXPECT_FLOAT_EQ(scene.player.stats_.armor.GetValue(), 0.0f);
  EXPECT_TRUE(scene.player.inventory_.empty());
  EXPECT_TRUE(scene.level_up_options.empty());
  EXPECT_TRUE(scene.item_options.empty());
}

}  // namespace
}  // namespace arelto
