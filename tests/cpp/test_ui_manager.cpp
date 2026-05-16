// tests/cpp/test_ui_manager.cpp
// Unit tests for UIManager behavior and screen composition.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "constants/chest.h"
#include "entity_manager.h"
#include "event_manager.h"
#include "items.h"
#include "render_resources.h"
#include "scene.h"
#include "test_helpers.h"
#include "types.h"
#include "ui/containers.h"
#include "ui/widgets.h"
#include "ui_manager.h"
#include "upgrades.h"

namespace arelto {
namespace {

UpgradeOptions MakeSpellUpgradeOptions() {
  UpgradeOptions options;
  std::vector<SpellStatModifier> fireball_stat_modifiers;
  fireball_stat_modifiers.push_back(SpellStatModifier{
      SpellUpgradeType::damage, ModifierType::flat, 2.0f,
      ValueRange{10.0f, 12.0f}, "", IsHigherBetter(SpellUpgradeType::damage)});

  auto fireball_option =
      SpellStatUpgrade{0, "Fireball", fireball_stat_modifiers, Size2D{60, 60}};

  std::vector<SpellStatModifier> frostbolt_stat_modifiers;
  frostbolt_stat_modifiers.push_back(SpellStatModifier{
      SpellUpgradeType::damage, ModifierType::flat, -0.5f,
      ValueRange{1.50f, 1.0f}, "", IsHigherBetter(SpellUpgradeType::cooldown)});

  auto frostbolt_option = SpellStatUpgrade{
      1, "Frostbolt", frostbolt_stat_modifiers, Size2D{100, 100}};

  options.push_back(std::make_unique<SpellStatUpgrade>(fireball_option));
  options.push_back(std::make_unique<SpellStatUpgrade>(frostbolt_option));
  return options;
}

UpgradeOptions MakeSingleSpellUpgradeOption() {
  UpgradeOptions options;
  std::vector<SpellStatModifier> fireball_stat_modifiers;
  SpellUpgradeType upgrade_type = SpellUpgradeType::damage;
  fireball_stat_modifiers.push_back(SpellStatModifier{
      upgrade_type, ModifierType::flat, 2.0f, ValueRange{10.0f, 12.0f},
      ResolveSpellUpgradeDescription(upgrade_type),
      IsHigherBetter(SpellUpgradeType::damage)});

  auto fireball_option =
      SpellStatUpgrade{0, "Fireball", fireball_stat_modifiers, Size2D{60, 60}};

  options.push_back(std::make_unique<SpellStatUpgrade>(fireball_option));
  return options;
}

std::unique_ptr<Upgrade> MakeArmorItemUpgrade() {
  std::vector<ItemStatModifier> stat_modifiers;
  stat_modifiers.push_back(
      ItemStatModifier{ItemUpgradeType::armor, ModifierType::flat, 1.0f,
                       ValueRange{1.0f, 2.0f}, "Increase Armor"});
  return std::make_unique<ItemUpgrade>(
      ItemId::elia_armor_plate, "Skewer-safe Armorplate of Elia",
      std::move(stat_modifiers), std::vector<ItemTriggerModifier>{}, "");
}

std::unique_ptr<Upgrade> MakeTriggerOnlyItemUpgrade() {
  std::vector<ItemTriggerModifier> trigger_modifiers;
  trigger_modifiers.push_back(ItemTriggerModifier{
      "Heal 5 HP on kill", std::make_unique<HealOnKillEffect>(5)});
  return std::make_unique<ItemUpgrade>(ItemId::damodei_claw, "Claw of Damodei",
                                       std::vector<ItemStatModifier>{},
                                       std::move(trigger_modifiers), "");
}

std::unique_ptr<Upgrade> MakeNegativeStatItemUpgrade() {
  std::vector<ItemStatModifier> stat_modifiers;
  stat_modifiers.push_back(ItemStatModifier{
      ItemUpgradeType::movement_speed, ModifierType::percent_mult, -0.05f,
      ValueRange{1.0f, 0.95f}, "Slow Movement", true});
  return std::make_unique<ItemUpgrade>(
      ItemId::elia_armor_plate, "Skewer-safe Armorplate of Elia",
      std::move(stat_modifiers), std::vector<ItemTriggerModifier>{}, "");
}

UpgradeOptions MakeItemUpgradeOptions() {
  UpgradeOptions options;
  options.push_back(MakeArmorItemUpgrade());
  options.push_back(MakeTriggerOnlyItemUpgrade());
  return options;
}

UpgradeOptions MakeTriggerOnlyItemUpgradeOptions() {
  UpgradeOptions options;
  options.push_back(MakeTriggerOnlyItemUpgrade());
  return options;
}

UpgradeOptions MakeNegativeItemUpgradeOptions() {
  UpgradeOptions options;
  options.push_back(MakeNegativeStatItemUpgrade());
  return options;
}

void ExpectColorEq(const SDL_Color& actual, const SDL_Color& expected) {
  EXPECT_EQ(actual.r, expected.r);
  EXPECT_EQ(actual.g, expected.g);
  EXPECT_EQ(actual.b, expected.b);
  EXPECT_EQ(actual.a, expected.a);
}

class UIManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = testing::CreateTestScene();
    ui_manager_.SetupUI(resources_, config_, event_manager_);
  }

  EventContext MakeEventContext() {
    return testing::MakeEventContext(scene_, event_manager_);
  }

  RenderResources resources_;
  UIConfig config_ = MakeDefaultUIConfig();
  UIManager ui_manager_;
  EventManager event_manager_;
  Scene scene_;
};

TEST_F(UIManagerTest, SetupUI_CreatesRootCanvas) {
  auto* root = ui_manager_.GetRootWidget();
  ASSERT_NE(root, nullptr);
  EXPECT_EQ(root->GetId(), "root");

  SDL_Rect bounds = root->GetComputedBounds();
  EXPECT_EQ(bounds.x, 0);
  EXPECT_EQ(bounds.y, 0);
  EXPECT_EQ(bounds.w, kWindowWidth);
  EXPECT_EQ(bounds.h, kWindowHeight);
}

TEST_F(UIManagerTest, SetupUI_BuildsHudWithExpectedDefaultState) {
  auto* top_left = ui_manager_.GetWidget<VBox>("hud_top_left");
  auto* bottom_left = ui_manager_.GetWidget<VBox>("hud_bottom_left");
  auto* timer_icon = ui_manager_.GetWidget<UIImage>("timer_icon");
  auto* timer_text = ui_manager_.GetWidget<UILabel>("timer_text");
  auto* level_icon = ui_manager_.GetWidget<UIImage>("level_icon");
  auto* level_text = ui_manager_.GetWidget<UILabel>("level_text");
  auto* health_bar = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  auto* health_text = ui_manager_.GetWidget<UILabel>("health_text");
  auto* exp_bar = ui_manager_.GetWidget<UIProgressBar>("exp_bar");
  auto* exp_text = ui_manager_.GetWidget<UILabel>("exp_text");

  ASSERT_NE(top_left, nullptr);
  ASSERT_NE(bottom_left, nullptr);
  ASSERT_NE(timer_icon, nullptr);
  ASSERT_NE(timer_text, nullptr);
  ASSERT_NE(level_icon, nullptr);
  ASSERT_NE(level_text, nullptr);
  ASSERT_NE(health_bar, nullptr);
  ASSERT_NE(health_text, nullptr);
  ASSERT_NE(exp_bar, nullptr);
  ASSERT_NE(exp_text, nullptr);

  EXPECT_FLOAT_EQ(health_bar->GetPercent(), 1.0f);
  EXPECT_FLOAT_EQ(exp_bar->GetPercent(), 0.0f);
  EXPECT_TRUE(timer_text->GetText().empty());
  EXPECT_TRUE(level_text->GetText().empty());
  EXPECT_TRUE(health_text->GetText().empty());
  EXPECT_TRUE(exp_text->GetText().empty());
}

TEST_F(UIManagerTest,
       SetupUI_BuildsSettingsMenuWithExpectedControlsAndDefaults) {
  auto* settings = ui_manager_.GetSettingsRoot();
  auto* title = ui_manager_.GetWidget<UILabel>("settings_title");
  auto* volume_label = ui_manager_.GetWidget<UILabel>("volume_label");
  auto* debug_label = ui_manager_.GetWidget<UILabel>("debug_label");
  auto* volume_slider = ui_manager_.GetWidget<UIProgressBar>("volume_slider");
  auto* mute_checkbox = ui_manager_.GetWidget<UICheckbox>("mute_checkbox");
  auto* occupancy_map_checkbox =
      ui_manager_.GetWidget<UICheckbox>("occupancy_map_checkbox");
  auto* ray_caster_checkbox =
      ui_manager_.GetWidget<UICheckbox>("ray_caster_checkbox");
  auto* resume_button = ui_manager_.GetWidget<UIButton>("resume_button");
  auto* main_menu_button = ui_manager_.GetWidget<UIButton>("main_menu_button");

  ASSERT_NE(settings, nullptr);
  ASSERT_NE(title, nullptr);
  ASSERT_NE(volume_label, nullptr);
  ASSERT_NE(debug_label, nullptr);
  ASSERT_NE(volume_slider, nullptr);
  ASSERT_NE(mute_checkbox, nullptr);
  ASSERT_NE(occupancy_map_checkbox, nullptr);
  ASSERT_NE(ray_caster_checkbox, nullptr);
  ASSERT_NE(resume_button, nullptr);
  ASSERT_NE(main_menu_button, nullptr);

  EXPECT_FALSE(settings->IsVisible());
  EXPECT_EQ(title->GetText(), "SETTINGS");
  EXPECT_EQ(volume_label->GetText(), "MUSIC VOLUME");
  EXPECT_EQ(debug_label->GetText(), "DEBUG");
  EXPECT_FLOAT_EQ(volume_slider->GetPercent(), 1.0f);
  EXPECT_FALSE(mute_checkbox->IsChecked());
  EXPECT_FALSE(occupancy_map_checkbox->IsChecked());
  EXPECT_FALSE(ray_caster_checkbox->IsChecked());
  EXPECT_EQ(resume_button->GetLabel(), "RESUME");
  EXPECT_EQ(main_menu_button->GetLabel(), "MAIN MENU");
}

TEST_F(UIManagerTest, SetupUI_BuildsStartScreenWithBeginButton) {
  auto* start_screen = ui_manager_.GetStartScreenRoot();
  ASSERT_NE(start_screen, nullptr);
  EXPECT_EQ(start_screen->GetId(), "start_screen");
  EXPECT_FALSE(start_screen->IsVisible());

  auto* begin_button = start_screen->FindWidgetAs<UIButton>("begin_button");
  ASSERT_NE(begin_button, nullptr);
}

TEST_F(UIManagerTest, SetupUI_BuildsGameOverScreenWithImageContainer) {
  auto* game_over = ui_manager_.GetGameOverScreenRoot();
  ASSERT_NE(game_over, nullptr);
  EXPECT_FALSE(game_over->IsVisible());

  auto* game_over_bar = game_over->FindWidgetAs<Panel>("game_over_bar");
  auto* game_over_image = game_over->FindWidgetAs<UIImage>("game_over_image");
  ASSERT_NE(game_over_bar, nullptr);
  ASSERT_NE(game_over_image, nullptr);
}

TEST_F(UIManagerTest, SetupUI_BuildsQuitConfirmMenuWithYesAndNoButtons) {
  auto* quit_confirm = ui_manager_.GetQuitConfirmRoot();
  ASSERT_NE(quit_confirm, nullptr);
  EXPECT_FALSE(quit_confirm->IsVisible());

  auto* yes_button = quit_confirm->FindWidgetAs<UIButton>("quit_yes_button");
  auto* no_button = quit_confirm->FindWidgetAs<UIButton>("quit_no_button");
  ASSERT_NE(yes_button, nullptr);
  ASSERT_NE(no_button, nullptr);
  EXPECT_EQ(yes_button->GetLabel(), "YES");
  EXPECT_EQ(no_button->GetLabel(), "NO");
}

TEST_F(UIManagerTest,
       SetupUI_BuildsChestOpeningOverlayWithConfiguredAnimation) {
  auto* chest_opening = ui_manager_.GetChestOpeningRoot();
  ASSERT_NE(chest_opening, nullptr);
  EXPECT_FALSE(chest_opening->IsVisible());

  auto* animation =
      chest_opening->FindWidgetAs<UIAnimation>("chest_animated_image");
  ASSERT_NE(animation, nullptr);
  EXPECT_EQ(animation->GetFrames().size(),
            static_cast<size_t>(kChestTotalAnimFrames));
  EXPECT_FLOAT_EQ(animation->GetFrameDuration(),
                  kChestAnimationFrameDuration / 1000.0f);
  EXPECT_FALSE(animation->GetIsLoop());
}

TEST_F(UIManagerTest, SetupUI_BuildsInventoryContainerHiddenByDefault) {
  auto* container = ui_manager_.GetWidget<Panel>("inventory_container");
  auto* inventory_bar = ui_manager_.GetWidget<HBox>("inventory_bar");
  ASSERT_NE(container, nullptr);
  ASSERT_NE(inventory_bar, nullptr);
  EXPECT_FALSE(container->IsVisible());
  EXPECT_TRUE(inventory_bar->GetChildren().empty());
}

TEST_F(UIManagerTest, UpdateTimer_SetsTimerLabelToWholeSeconds) {
  ui_manager_.UpdateTimer(123.8f);

  auto* timer_text = ui_manager_.GetWidget<UILabel>("timer_text");
  ASSERT_NE(timer_text, nullptr);
  EXPECT_EQ(timer_text->GetText(), "123");
}

TEST_F(UIManagerTest, UpdateSettingsMenu_SyncsVolumeAndCheckboxState) {
  GameStatus game_status{};
  game_status.is_headless = true;
  game_status.show_occupancy_map = true;
  game_status.show_ray_caster = false;

  ui_manager_.UpdateSettingsMenu(64.0f, true, game_status);

  auto* volume_slider = ui_manager_.GetWidget<UIProgressBar>("volume_slider");
  auto* mute_checkbox = ui_manager_.GetWidget<UICheckbox>("mute_checkbox");
  auto* occupancy_map_checkbox =
      ui_manager_.GetWidget<UICheckbox>("occupancy_map_checkbox");
  auto* ray_caster_checkbox =
      ui_manager_.GetWidget<UICheckbox>("ray_caster_checkbox");

  ASSERT_NE(volume_slider, nullptr);
  ASSERT_NE(mute_checkbox, nullptr);
  ASSERT_NE(occupancy_map_checkbox, nullptr);
  ASSERT_NE(ray_caster_checkbox, nullptr);

  EXPECT_FLOAT_EQ(volume_slider->GetPercent(), 0.5f);
  EXPECT_TRUE(mute_checkbox->IsChecked());
  EXPECT_TRUE(occupancy_map_checkbox->IsChecked());
  EXPECT_FALSE(ray_caster_checkbox->IsChecked());
}

TEST_F(UIManagerTest, UpdateSettingsMenu_ClampsSliderViaProgressBarPercent) {
  GameStatus game_status{};
  game_status.is_headless = true;

  ui_manager_.UpdateSettingsMenu(512.0f, false, game_status);
  auto* volume_slider = ui_manager_.GetWidget<UIProgressBar>("volume_slider");
  ASSERT_NE(volume_slider, nullptr);
  EXPECT_FLOAT_EQ(volume_slider->GetPercent(), 1.0f);

  ui_manager_.UpdateSettingsMenu(-32.0f, false, game_status);
  EXPECT_FLOAT_EQ(volume_slider->GetPercent(), 0.0f);
}

TEST_F(UIManagerTest, PlayerDamagedEvent_UpdatesHealthBarAndHealthText) {
  scene_.player.stats_.health = 75;
  scene_.player.stats_.max_health.SetBaseValue(100.0f);

  auto event_context = MakeEventContext();
  event_manager_.DispatchImmediate(PlayerDamagedEvent{0, 25}, event_context);

  auto* health_bar = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  auto* health_text = ui_manager_.GetWidget<UILabel>("health_text");
  ASSERT_NE(health_bar, nullptr);
  ASSERT_NE(health_text, nullptr);
  EXPECT_FLOAT_EQ(health_bar->GetPercent(), 0.75f);
  EXPECT_EQ(health_text->GetText(), "75/100");
}

TEST_F(UIManagerTest, PlayerHealedEvent_UpdatesHealthBarAndHealthText) {
  scene_.player.stats_.health = 85;
  scene_.player.stats_.max_health.SetBaseValue(100.0f);

  auto event_context = MakeEventContext();
  event_manager_.DispatchImmediate(PlayerHealedEvent{10}, event_context);

  auto* health_bar = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  auto* health_text = ui_manager_.GetWidget<UILabel>("health_text");
  ASSERT_NE(health_bar, nullptr);
  ASSERT_NE(health_text, nullptr);
  EXPECT_FLOAT_EQ(health_bar->GetPercent(), 0.85f);
  EXPECT_EQ(health_text->GetText(), "85/100");
}

TEST_F(UIManagerTest, ExpGemCollectedEvent_UpdatesExpBarAndExpText) {
  scene_.player.stats_.exp_points = 500;
  scene_.player.stats_.exp_points_required.SetBaseValue(1000.0f);

  auto event_context = MakeEventContext();
  event_manager_.DispatchImmediate(ExpGemCollectedEvent{0, 200}, event_context);

  auto* exp_bar = ui_manager_.GetWidget<UIProgressBar>("exp_bar");
  auto* exp_text = ui_manager_.GetWidget<UILabel>("exp_text");
  ASSERT_NE(exp_bar, nullptr);
  ASSERT_NE(exp_text, nullptr);
  EXPECT_FLOAT_EQ(exp_bar->GetPercent(), 0.5f);
  EXPECT_EQ(exp_text->GetText(), "500/1000");
}

TEST_F(UIManagerTest, PlayerLevelUpEvent_UpdatesExpTextAndLevelText) {
  scene_.player.stats_.exp_points = 0;
  scene_.player.stats_.exp_points_required.SetBaseValue(2000.0f);
  scene_.player.stats_.level = 5;

  auto event_context = MakeEventContext();
  event_manager_.DispatchImmediate(PlayerLevelUpEvent{}, event_context);

  auto* exp_text = ui_manager_.GetWidget<UILabel>("exp_text");
  auto* level_text = ui_manager_.GetWidget<UILabel>("level_text");
  ASSERT_NE(exp_text, nullptr);
  ASSERT_NE(level_text, nullptr);
  EXPECT_EQ(exp_text->GetText(), "0/2000");
  EXPECT_EQ(level_text->GetText(), "5");
}

TEST_F(UIManagerTest, SceneResetEvent_RefreshesHudAndInventoryFromScene) {
  scene_.player.stats_.exp_points = 250;
  scene_.player.stats_.exp_points_required.SetBaseValue(1000.0f);
  scene_.player.stats_.level = 3;
  scene_.player.stats_.health = 60;
  scene_.player.stats_.max_health.SetBaseValue(120.0f);
  scene_.player.inventory_ = {{ItemId::elia_armor_plate, 4}};

  auto event_context = MakeEventContext();
  event_manager_.DispatchImmediate(SceneResetEvent{}, event_context);

  auto* exp_bar = ui_manager_.GetWidget<UIProgressBar>("exp_bar");
  auto* exp_text = ui_manager_.GetWidget<UILabel>("exp_text");
  auto* level_text = ui_manager_.GetWidget<UILabel>("level_text");
  auto* health_bar = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  auto* health_text = ui_manager_.GetWidget<UILabel>("health_text");
  auto* inventory_container =
      ui_manager_.GetWidget<Panel>("inventory_container");
  auto* inventory_bar = ui_manager_.GetWidget<HBox>("inventory_bar");
  ASSERT_NE(exp_bar, nullptr);
  ASSERT_NE(exp_text, nullptr);
  ASSERT_NE(level_text, nullptr);
  ASSERT_NE(health_bar, nullptr);
  ASSERT_NE(health_text, nullptr);
  ASSERT_NE(inventory_container, nullptr);
  ASSERT_NE(inventory_bar, nullptr);

  EXPECT_FLOAT_EQ(exp_bar->GetPercent(), 0.25f);
  EXPECT_EQ(exp_text->GetText(), "250/1000");
  EXPECT_EQ(level_text->GetText(), "3");
  EXPECT_FLOAT_EQ(health_bar->GetPercent(), 0.5f);
  EXPECT_EQ(health_text->GetText(), "60/120");
  EXPECT_TRUE(inventory_container->IsVisible());
  ASSERT_EQ(inventory_bar->GetChildren().size(), 1u);

  auto* inventory_item =
      inventory_bar->FindWidgetAs<UIInventoryItem>("inventory_item_0");
  ASSERT_NE(inventory_item, nullptr);
  EXPECT_EQ(inventory_item->GetItemCount(), 4);
}

TEST_F(UIManagerTest, PlayerClaimedItemEvent_BuildsInventoryAndShowsContainer) {
  scene_.player.inventory_ = {{ItemId::elia_armor_plate, 2},
                              {ItemId::damodei_claw, 1}};

  auto event_context = MakeEventContext();
  event_manager_.DispatchImmediate(PlayerClaimedItemEvent{}, event_context);

  auto* inventory_container =
      ui_manager_.GetWidget<Panel>("inventory_container");
  auto* inventory_bar = ui_manager_.GetWidget<HBox>("inventory_bar");
  ASSERT_NE(inventory_container, nullptr);
  ASSERT_NE(inventory_bar, nullptr);
  EXPECT_TRUE(inventory_container->IsVisible());
  ASSERT_EQ(inventory_bar->GetChildren().size(), 2u);

  auto* item_0 =
      inventory_bar->FindWidgetAs<UIInventoryItem>("inventory_item_0");
  auto* multiplier =
      inventory_bar->FindWidgetAs<UILabel>("inventory_item_0_multiplier");
  ASSERT_NE(item_0, nullptr);
  ASSERT_NE(multiplier, nullptr);
  EXPECT_EQ(item_0->GetItemId(), ItemId::elia_armor_plate);
  EXPECT_EQ(item_0->GetItemCount(), 2);
  EXPECT_EQ(multiplier->GetText(), "x2");
}

TEST_F(UIManagerTest, PlayerClaimedItemEvent_UpdatesExistingInventoryCounts) {
  scene_.player.inventory_ = {{ItemId::elia_armor_plate, 2}};

  auto event_context = MakeEventContext();
  event_manager_.DispatchImmediate(PlayerClaimedItemEvent{}, event_context);

  auto* inventory_bar = ui_manager_.GetWidget<HBox>("inventory_bar");
  ASSERT_NE(inventory_bar, nullptr);
  auto* item_before =
      inventory_bar->FindWidgetAs<UIInventoryItem>("inventory_item_0");
  ASSERT_NE(item_before, nullptr);

  scene_.player.inventory_ = {{ItemId::elia_armor_plate, 5}};
  event_manager_.DispatchImmediate(PlayerClaimedItemEvent{}, event_context);

  auto* item_after =
      inventory_bar->FindWidgetAs<UIInventoryItem>("inventory_item_0");
  auto* multiplier =
      inventory_bar->FindWidgetAs<UILabel>("inventory_item_0_multiplier");
  ASSERT_NE(item_after, nullptr);
  ASSERT_NE(multiplier, nullptr);
  EXPECT_EQ(item_after, item_before);
  EXPECT_EQ(item_after->GetItemCount(), 5);
  EXPECT_EQ(multiplier->GetText(), "x5");
}

TEST_F(
    UIManagerTest,
    PlayerClaimedItemEvent_RemovesStaleInventoryWidgetsWhenInventoryShrinks) {
  scene_.player.inventory_ = {{ItemId::elia_armor_plate, 2},
                              {ItemId::damodei_claw, 1}};

  auto event_context = MakeEventContext();
  event_manager_.DispatchImmediate(PlayerClaimedItemEvent{}, event_context);

  auto* inventory_bar = ui_manager_.GetWidget<HBox>("inventory_bar");
  ASSERT_NE(inventory_bar, nullptr);
  ASSERT_EQ(inventory_bar->GetChildren().size(), 2u);

  scene_.player.inventory_ = {{ItemId::elia_armor_plate, 2}};
  event_manager_.DispatchImmediate(PlayerClaimedItemEvent{}, event_context);

  EXPECT_EQ(inventory_bar->GetChildren().size(), 1u);
  EXPECT_NE(inventory_bar->FindWidget("inventory_item_0"), nullptr);
  EXPECT_EQ(inventory_bar->FindWidget("inventory_item_1"), nullptr);
}

TEST_F(UIManagerTest,
       SceneResetEvent_HidesInventoryContainerWhenInventoryIsEmpty) {
  scene_.player.inventory_ = {{ItemId::elia_armor_plate, 2}};

  auto event_context = MakeEventContext();
  event_manager_.DispatchImmediate(SceneResetEvent{}, event_context);

  auto* inventory_container =
      ui_manager_.GetWidget<Panel>("inventory_container");
  auto* inventory_bar = ui_manager_.GetWidget<HBox>("inventory_bar");
  ASSERT_NE(inventory_container, nullptr);
  ASSERT_NE(inventory_bar, nullptr);
  EXPECT_TRUE(inventory_container->IsVisible());

  scene_.player.inventory_.clear();
  event_manager_.DispatchImmediate(SceneResetEvent{}, event_context);

  EXPECT_FALSE(inventory_container->IsVisible());
  EXPECT_TRUE(inventory_bar->GetChildren().empty());
}

TEST_F(UIManagerTest, DispatchingUnsubscribedEvent_LeavesHudStateUnchanged) {
  scene_.player.stats_.health = 75;
  scene_.player.stats_.max_health.SetBaseValue(100.0f);

  auto event_context = MakeEventContext();
  event_manager_.DispatchImmediate(PlayerDamagedEvent{0, 25}, event_context);

  auto* health_bar = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  auto* health_text = ui_manager_.GetWidget<UILabel>("health_text");
  ASSERT_NE(health_bar, nullptr);
  ASSERT_NE(health_text, nullptr);
  float health_percent_before = health_bar->GetPercent();
  std::string health_text_before = health_text->GetText();

  event_manager_.DispatchImmediate(EnemyKilledEvent{0}, event_context);

  EXPECT_FLOAT_EQ(health_bar->GetPercent(), health_percent_before);
  EXPECT_EQ(health_text->GetText(), health_text_before);
}

TEST_F(UIManagerTest, BuildLevelUpMenu_CreatesCardPerOption) {
  ui_manager_.BuildLevelUpMenu(MakeSpellUpgradeOptions());

  auto* level_up_menu = ui_manager_.GetLevelUpRoot();
  ASSERT_NE(level_up_menu, nullptr);
  EXPECT_FALSE(level_up_menu->IsVisible());

  auto* cards = level_up_menu->FindWidgetAs<HBox>("level_up_cards");
  ASSERT_NE(cards, nullptr);
  EXPECT_EQ(cards->GetChildren().size(), 2u);
}

TEST_F(UIManagerTest, BuildLevelUpMenu_ReplacesExistingMenu) {
  ui_manager_.BuildLevelUpMenu(MakeSpellUpgradeOptions());
  auto* level_up_menu = ui_manager_.GetLevelUpRoot();
  ASSERT_NE(level_up_menu, nullptr);
  ASSERT_NE(level_up_menu->FindWidget("level_up_card_1"), nullptr);

  ui_manager_.BuildLevelUpMenu(MakeSingleSpellUpgradeOption());
  level_up_menu = ui_manager_.GetLevelUpRoot();
  ASSERT_NE(level_up_menu, nullptr);

  auto* cards = level_up_menu->FindWidgetAs<HBox>("level_up_cards");
  ASSERT_NE(cards, nullptr);
  EXPECT_EQ(cards->GetChildren().size(), 1u);
  EXPECT_EQ(level_up_menu->FindWidget("level_up_card_1"), nullptr);
}

TEST_F(UIManagerTest,
       BuildLevelUpMenu_RendersNameDescriptionStatsAndSelectButton) {
  resources_.projectiles.resize(2, reinterpret_cast<SDL_Texture*>(0x1));
  ui_manager_.BuildLevelUpMenu(MakeSingleSpellUpgradeOption());

  auto* level_up_menu = ui_manager_.GetLevelUpRoot();
  ASSERT_NE(level_up_menu, nullptr);

  auto* icon = level_up_menu->FindWidgetAs<UIImage>("level_up_card_0_icon");
  auto* name = level_up_menu->FindWidgetAs<UILabel>("level_up_card_0_name");
  auto* description =
      level_up_menu->FindWidgetAs<UILabel>("level_up_card_0_desc_0");
  auto* stats = level_up_menu->FindWidgetAs<UILabel>("level_up_card_0_stats_0");
  auto* select_button =
      level_up_menu->FindWidgetAs<UIButton>("select_button_0");
  ASSERT_NE(icon, nullptr);
  ASSERT_NE(name, nullptr);
  ASSERT_NE(description, nullptr);
  ASSERT_NE(stats, nullptr);
  ASSERT_NE(select_button, nullptr);

  EXPECT_EQ(name->GetText(), "Fireball");
  EXPECT_EQ(description->GetText(), "Increase Spell Damage");
  EXPECT_EQ(stats->GetText(), "10.00 -> 12.00");
  EXPECT_EQ(select_button->GetLabel(), "SELECT");
}

TEST_F(UIManagerTest, BuildItemMenu_CreatesCardPerOption) {
  ui_manager_.BuildItemMenu(MakeItemUpgradeOptions());

  auto* item_menu = ui_manager_.GetItemMenuRoot();
  ASSERT_NE(item_menu, nullptr);
  EXPECT_FALSE(item_menu->IsVisible());

  auto* cards = item_menu->FindWidgetAs<HBox>("item_cards");
  ASSERT_NE(cards, nullptr);
  EXPECT_EQ(cards->GetChildren().size(), 2u);
}

TEST_F(UIManagerTest, BuildItemMenu_ReplacesExistingMenu) {
  ui_manager_.BuildItemMenu(MakeItemUpgradeOptions());
  auto* item_menu = ui_manager_.GetItemMenuRoot();
  ASSERT_NE(item_menu, nullptr);
  ASSERT_NE(item_menu->FindWidget("item_card_1"), nullptr);

  ui_manager_.BuildItemMenu(MakeTriggerOnlyItemUpgradeOptions());
  item_menu = ui_manager_.GetItemMenuRoot();
  ASSERT_NE(item_menu, nullptr);

  auto* cards = item_menu->FindWidgetAs<HBox>("item_cards");
  ASSERT_NE(cards, nullptr);
  EXPECT_EQ(cards->GetChildren().size(), 1u);
  EXPECT_EQ(item_menu->FindWidget("item_card_1"), nullptr);
}

TEST_F(UIManagerTest, BuildItemMenu_RendersTriggerOnlyRowsWithoutStatsLabel) {
  ui_manager_.BuildItemMenu(MakeTriggerOnlyItemUpgradeOptions());

  auto* item_menu = ui_manager_.GetItemMenuRoot();
  ASSERT_NE(item_menu, nullptr);

  auto* description = item_menu->FindWidgetAs<UILabel>("item_card_0_desc_0");
  ASSERT_NE(description, nullptr);
  EXPECT_EQ(description->GetText(), "Heal 5 HP on kill");
  EXPECT_EQ(item_menu->FindWidget("item_card_0_stats_0"), nullptr);
}

TEST_F(UIManagerTest, BuildItemMenu_UsesClaimButtonLabel) {
  ui_manager_.BuildItemMenu(MakeItemUpgradeOptions());

  auto* item_menu = ui_manager_.GetItemMenuRoot();
  ASSERT_NE(item_menu, nullptr);

  auto* claim_button = item_menu->FindWidgetAs<UIButton>("select_button_0");
  ASSERT_NE(claim_button, nullptr);
  EXPECT_EQ(claim_button->GetLabel(), "CLAIM");
}

TEST_F(UIManagerTest, BuildItemMenu_UsesNegativeColorForDecreasingStatRow) {
  ui_manager_.BuildItemMenu(MakeNegativeItemUpgradeOptions());

  auto* item_menu = ui_manager_.GetItemMenuRoot();
  ASSERT_NE(item_menu, nullptr);

  auto* description = item_menu->FindWidgetAs<UILabel>("item_card_0_desc_0");
  auto* stats = item_menu->FindWidgetAs<UILabel>("item_card_0_stats_0");
  ASSERT_NE(description, nullptr);
  ASSERT_NE(stats, nullptr);

  EXPECT_EQ(description->GetText(), "Slow Movement");
  EXPECT_EQ(stats->GetText(), "1.00 -> 0.95");
  ExpectColorEq(stats->GetColor(), MakeDefaultUIConfig().colors.negative_red);
}

class UIManagerEntityManagerIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = testing::CreateTestScene();
    entity_manager_.Initialize(event_manager_);
    ui_manager_.SetupUI(resources_, config_, event_manager_);
  }

  EventContext MakeEventContext() {
    return testing::MakeEventContext(scene_, event_manager_);
  }

  RenderResources resources_;
  UIConfig config_ = MakeDefaultUIConfig();
  UIManager ui_manager_;
  EntityManager entity_manager_;
  EventManager event_manager_;
  Scene scene_;
};

TEST_F(UIManagerEntityManagerIntegrationTest,
       PlayerDamageAndHealEvents_UpdateSceneAndHudConsistently) {
  scene_.player.stats_.health = 100;
  scene_.player.stats_.max_health.SetBaseValue(100.0f);
  scene_.player.stats_.armor.SetBaseValue(0.0f);
  scene_.player.is_alive_ = true;
  scene_.player.is_invulnerable = false;

  auto event_context = MakeEventContext();

  event_manager_.DispatchImmediate(PlayerDamagedEvent{0, 25}, event_context);

  auto* health_bar = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  auto* health_text = ui_manager_.GetWidget<UILabel>("health_text");
  ASSERT_NE(health_bar, nullptr);
  ASSERT_NE(health_text, nullptr);

  EXPECT_EQ(scene_.player.stats_.health, 75);
  EXPECT_FLOAT_EQ(health_bar->GetPercent(), 0.75f);
  EXPECT_EQ(health_text->GetText(), "75/100");

  scene_.player.TakeHealing(10);
  event_manager_.DispatchImmediate(PlayerHealedEvent{10}, event_context);

  EXPECT_EQ(scene_.player.stats_.health, 85);
  EXPECT_FLOAT_EQ(health_bar->GetPercent(), 0.85f);
  EXPECT_EQ(health_text->GetText(), "85/100");
}

}  // namespace
}  // namespace arelto
