// tests/cpp/test_item_manager.cpp
// Unit tests for ItemManager, ItemTriggerEffect, and ItemUpgrade pickup flow.

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "entity_manager.h"
#include "event_manager.h"
#include "item_manager.h"
#include "items.h"
#include "scene.h"
#include "test_helpers.h"

namespace arelto {
namespace {

class ItemManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = testing::CreateTestScene();
    entity_manager_.Initialize(event_manager_);
    item_manager_.Initialize(event_manager_);
    // Drop the player below max HP so TakeHealing is observable. TakeHealing
    // clamps to max_health, so a fully-healed player cannot be healed further.
    scene_.player.stats_.health -= kHealingHeadroom;
  }

  static constexpr int kHealingHeadroom = 50;

  EventContext MakeEventContext() {
    return testing::MakeEventContext(scene_, event_manager_);
  }

  void EmitEnemyKilledAndDispatch() {
    event_manager_.Emit(EnemyKilledEvent{0});
    auto event_context = MakeEventContext();
    event_manager_.Dispatch(event_context);
  }

  Scene scene_;
  EventManager event_manager_;
  EntityManager entity_manager_;
  ItemManager item_manager_;
};

// =============================================================================
// ItemTriggerEffect routing through ItemManager
// =============================================================================

TEST_F(ItemManagerTest, HealOnKillEffect_HealsPlayerOnEnemyKilled) {
  int initial_health = scene_.player.stats_.health;
  item_manager_.AddItem(std::make_unique<HealOnKillEffect>(5));

  EmitEnemyKilledAndDispatch();

  EXPECT_EQ(scene_.player.stats_.health, initial_health + 5);
}

TEST_F(ItemManagerTest, HealOnKillEffect_NoHeal_OnOtherEvents) {
  int initial_health = scene_.player.stats_.health;
  item_manager_.AddItem(std::make_unique<HealOnKillEffect>(5));

  event_manager_.Emit(ChestOpenedEvent{0});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_EQ(scene_.player.stats_.health, initial_health);
}

TEST_F(ItemManagerTest, MultipleHealOnKillEffects_BothFire) {
  int initial_health = scene_.player.stats_.health;
  item_manager_.AddItem(std::make_unique<HealOnKillEffect>(3));
  item_manager_.AddItem(std::make_unique<HealOnKillEffect>(7));

  EmitEnemyKilledAndDispatch();

  EXPECT_EQ(scene_.player.stats_.health, initial_health + 3 + 7);
}

TEST_F(ItemManagerTest, Clear_StopsDeliveringToClearedItems) {
  int initial_health = scene_.player.stats_.health;
  item_manager_.AddItem(std::make_unique<HealOnKillEffect>(5));

  item_manager_.RemoveAllItems();

  EmitEnemyKilledAndDispatch();

  EXPECT_EQ(scene_.player.stats_.health, initial_health);
}

// =============================================================================
// ItemUpgrade::Apply - composition of stat modifiers and trigger effects
// =============================================================================

TEST_F(ItemManagerTest,
       Apply_StatOnlyItem_UpdatesPlayerStatAndLeavesManagerEmpty) {
  float initial_armor = scene_.player.stats_.armor.GetValue();

  std::vector<ItemStatModifier> stat_modifiers;
  stat_modifiers.push_back(ItemStatModifier{
      ItemUpgradeType::armor, ModifierType::flat, 2.0f,
      ValueRange{initial_armor, initial_armor + 2.0f}, "Increase Armor"});
  ItemUpgrade stat_only_item(ItemId::elia_armor_plate, "Armor",
                             std::move(stat_modifiers), {});

  stat_only_item.Apply(scene_.player, item_manager_);

  EXPECT_FLOAT_EQ(scene_.player.stats_.armor.GetValue(), initial_armor + 2.0f);

  int initial_health = scene_.player.stats_.health;
  EmitEnemyKilledAndDispatch();
  EXPECT_EQ(scene_.player.stats_.health, initial_health);
}

TEST_F(ItemManagerTest,
       Apply_TriggerOnlyItem_RegistersTriggerAndLeavesStatsUnchanged) {
  float initial_armor = scene_.player.stats_.armor.GetValue();
  int initial_health = scene_.player.stats_.health;

  std::vector<ItemTriggerModifier> trigger_modifiers;
  trigger_modifiers.push_back(ItemTriggerModifier{
      "Heal 5 HP on kill", std::make_unique<HealOnKillEffect>(5)});
  ItemUpgrade trigger_only_item(ItemId::damodei_claw, "Claw", {},
                                std::move(trigger_modifiers));

  trigger_only_item.Apply(scene_.player, item_manager_);

  EXPECT_FLOAT_EQ(scene_.player.stats_.armor.GetValue(), initial_armor);

  EmitEnemyKilledAndDispatch();
  EXPECT_EQ(scene_.player.stats_.health, initial_health + 5);
}

TEST_F(ItemManagerTest, Apply_HybridItem_AppliesStatAndRegistersTrigger) {
  float initial_armor = scene_.player.stats_.armor.GetValue();
  int initial_health = scene_.player.stats_.health;

  std::vector<ItemStatModifier> stat_modifiers;
  stat_modifiers.push_back(ItemStatModifier{
      ItemUpgradeType::armor, ModifierType::flat, 1.0f,
      ValueRange{initial_armor, initial_armor + 1.0f}, "Increase Armor"});
  std::vector<ItemTriggerModifier> trigger_modifiers;
  trigger_modifiers.push_back(ItemTriggerModifier{
      "Heal 4 HP on kill", std::make_unique<HealOnKillEffect>(4)});
  ItemUpgrade hybrid_item(ItemId::elia_armor_plate, "Hybrid",
                          std::move(stat_modifiers),
                          std::move(trigger_modifiers));

  hybrid_item.Apply(scene_.player, item_manager_);

  EXPECT_FLOAT_EQ(scene_.player.stats_.armor.GetValue(), initial_armor + 1.0f);

  EmitEnemyKilledAndDispatch();
  EXPECT_EQ(scene_.player.stats_.health, initial_health + 4);
}

}  // namespace
}  // namespace arelto
