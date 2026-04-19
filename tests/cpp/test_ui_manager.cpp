// tests/cpp/test_ui_manager.cpp
// Unit tests for UIManager with widget tree

#include <gtest/gtest.h>

#include "entity_manager.h"
#include "event_manager.h"
#include "scene.h"
#include "test_helpers.h"
#include "ui/containers.h"
#include "ui/widget.h"
#include "ui/widgets.h"
#include "ui_manager.h"

namespace arelto {
namespace {

// =============================================================================
// UIWidget Hierarchy Tests
// =============================================================================

TEST(UIWidgetTest, AddChild_SetsParentAndGrowsChildren) {
  auto parent = std::make_shared<UIWidget>();
  auto child = std::make_shared<UIWidget>();

  parent->AddChild(child);

  EXPECT_EQ(child->GetParent(), parent.get());
  EXPECT_EQ(parent->GetChildren().size(), 1u);
}

TEST(UIWidgetTest, FindWidget_ById) {
  auto root = std::make_shared<UIWidget>();
  root->SetId("root");

  auto child1 = std::make_shared<UIWidget>();
  child1->SetId("child1");

  auto child2 = std::make_shared<UIWidget>();
  child2->SetId("child2");

  auto grandchild = std::make_shared<UIWidget>();
  grandchild->SetId("grandchild");

  root->AddChild(child1);
  root->AddChild(child2);
  child2->AddChild(grandchild);

  EXPECT_EQ(root->FindWidget("child1"), child1.get());
  EXPECT_EQ(root->FindWidget("grandchild"), grandchild.get());
  EXPECT_EQ(root->FindWidget("nonexistent"), nullptr);
}

TEST(UIWidgetTest, FindWidgetAs_ReturnsTypedPointer) {
  auto root = std::make_shared<UIWidget>();
  auto bar = std::make_shared<UIProgressBar>();
  bar->SetId("my_bar");
  root->AddChild(bar);

  auto* found = root->FindWidgetAs<UIProgressBar>("my_bar");
  ASSERT_NE(found, nullptr);
  found->SetPercent(0.5f);
  EXPECT_FLOAT_EQ(found->GetPercent(), 0.5f);
}

TEST(UIWidgetTest, Visibility_DefaultTrue) {
  UIWidget w;
  EXPECT_TRUE(w.IsVisible());
}

// =============================================================================
// VBox Layout Tests
// =============================================================================

TEST(VBoxTest, StacksChildrenVertically) {
  auto vbox = std::make_shared<VBox>();
  vbox->SetPosition(10, 20);
  vbox->SetSize(200, 400);
  vbox->SetSpacing(5);

  auto c1 = std::make_shared<UIWidget>();
  c1->SetSize(100, 30);
  auto c2 = std::make_shared<UIWidget>();
  c2->SetSize(100, 40);
  auto c3 = std::make_shared<UIWidget>();
  c3->SetSize(100, 50);

  vbox->AddChild(c1);
  vbox->AddChild(c2);
  vbox->AddChild(c3);

  vbox->ComputeLayout(0, 0, 800, 600);

  SDL_Rect b1 = c1->GetComputedBounds();
  SDL_Rect b2 = c2->GetComputedBounds();
  SDL_Rect b3 = c3->GetComputedBounds();

  // c1 starts at vbox position
  EXPECT_EQ(b1.y, 20);
  // c2 starts after c1 + spacing
  EXPECT_EQ(b2.y, 20 + 30 + 5);
  // c3 starts after c2 + spacing
  EXPECT_EQ(b3.y, 20 + 30 + 5 + 40 + 5);
}

// =============================================================================
// HBox Layout Tests
// =============================================================================

TEST(HBoxTest, ArrangesChildrenHorizontally) {
  auto hbox = std::make_shared<HBox>();
  hbox->SetPosition(10, 20);
  hbox->SetSize(400, 100);
  hbox->SetSpacing(10);

  auto c1 = std::make_shared<UIWidget>();
  c1->SetSize(60, 30);
  auto c2 = std::make_shared<UIWidget>();
  c2->SetSize(80, 30);

  hbox->AddChild(c1);
  hbox->AddChild(c2);

  hbox->ComputeLayout(0, 0, 800, 600);

  SDL_Rect b1 = c1->GetComputedBounds();
  SDL_Rect b2 = c2->GetComputedBounds();

  EXPECT_EQ(b1.x, 10);
  EXPECT_EQ(b2.x, 10 + 60 + 10);
}

// =============================================================================
// ProgressBar Tests
// =============================================================================

TEST(UIProgressBarTest, SetPercent_ClampsValues) {
  UIProgressBar bar;
  bar.SetPercent(1.5f);
  EXPECT_FLOAT_EQ(bar.GetPercent(), 1.0f);
  bar.SetPercent(-0.5f);
  EXPECT_FLOAT_EQ(bar.GetPercent(), 0.0f);
}

TEST(UIProgressBarTest, ClippedFillSrcRect_ScalesWithPercent) {
  UIProgressBar bar;
  bar.SetFillSrcRect({0, 0, 200, 30});
  bar.SetMaxFillSize(200, 30);

  bar.SetPercent(0.5f);
  SDL_Rect clipped = bar.GetClippedFillSrcRect();
  EXPECT_EQ(clipped.w, 100);  // 200 * 0.5

  bar.SetPercent(1.0f);
  clipped = bar.GetClippedFillSrcRect();
  EXPECT_EQ(clipped.w, 200);
}

// =============================================================================
// UIButton Tests
// =============================================================================

TEST(UIButtonTest, HoverState_ChangesCurrentSrcRect) {
  UIButton btn;
  btn.SetNormalSrcRect({0, 0, 100, 40});
  btn.SetHoverSrcRect({0, 40, 100, 40});

  btn.SetHovered(false);
  SDL_Rect normal = btn.GetCurrentSrcRect();
  EXPECT_EQ(normal.y, 0);

  btn.SetHovered(true);
  SDL_Rect hovered = btn.GetCurrentSrcRect();
  EXPECT_EQ(hovered.y, 40);
}

// =============================================================================
// UIManager Structure Tests (verify widget tree construction)
// =============================================================================

class UIManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    resources_ = {};
    EventManager dummy_event_manager;
    ui_manager_.SetupUI(resources_, dummy_event_manager);
  }

  UIResources resources_;
  UIManager ui_manager_;
};

TEST_F(UIManagerTest, SetupUI_CreatesRootWidget) {
  ASSERT_NE(ui_manager_.GetRootWidget(), nullptr);
}

TEST_F(UIManagerTest, BuildHUD_CreatesHealthBar) {
  auto* bar = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  ASSERT_NE(bar, nullptr);
}

TEST_F(UIManagerTest, BuildHUD_CreatesExpBar) {
  auto* bar = ui_manager_.GetWidget<UIProgressBar>("exp_bar");
  ASSERT_NE(bar, nullptr);
}

TEST_F(UIManagerTest, BuildHUD_CreatesTimerText) {
  auto* label = ui_manager_.GetWidget<UILabel>("timer_text");
  ASSERT_NE(label, nullptr);
}

TEST_F(UIManagerTest, BuildHUD_CreatesLevelText) {
  auto* label = ui_manager_.GetWidget<UILabel>("level_text");
  ASSERT_NE(label, nullptr);
}

TEST_F(UIManagerTest, BuildHUD_CreatesHealthText) {
  auto* label = ui_manager_.GetWidget<UILabel>("health_text");
  ASSERT_NE(label, nullptr);
}

TEST_F(UIManagerTest, BuildHUD_CreatesExpText) {
  auto* label = ui_manager_.GetWidget<UILabel>("exp_text");
  ASSERT_NE(label, nullptr);
}

TEST_F(UIManagerTest, BuildSettingsMenu_CreatesSettingsRoot) {
  auto* settings = ui_manager_.GetSettingsRoot();
  ASSERT_NE(settings, nullptr);
}

TEST_F(UIManagerTest, BuildSettingsMenu_CreatesMuteCheckbox) {
  auto* checkbox = ui_manager_.GetWidget<UICheckbox>("mute_checkbox");
  ASSERT_NE(checkbox, nullptr);
}

TEST_F(UIManagerTest, BuildSettingsMenu_CreatesResumeButton) {
  auto* btn = ui_manager_.GetWidget<UIButton>("resume_button");
  ASSERT_NE(btn, nullptr);
  EXPECT_EQ(btn->GetLabel(), "RESUME");
}

TEST_F(UIManagerTest, BuildSettingsMenu_CreatesVolumeSlider) {
  auto* slider = ui_manager_.GetWidget<UIProgressBar>("volume_slider");
  ASSERT_NE(slider, nullptr);
}

TEST_F(UIManagerTest, BuildSettingsMenu_CreatesOccupancyMapCheckbox) {
  auto* checkbox = ui_manager_.GetWidget<UICheckbox>("occupancy_map_checkbox");
  ASSERT_NE(checkbox, nullptr);
}

TEST_F(UIManagerTest, BuildSettingsMenu_CreatesRayCasterCheckbox) {
  auto* checkbox = ui_manager_.GetWidget<UICheckbox>("ray_caster_checkbox");
  ASSERT_NE(checkbox, nullptr);
}

TEST_F(UIManagerTest, SettingsMenu_StartsHidden) {
  auto* settings = ui_manager_.GetSettingsRoot();
  ASSERT_NE(settings, nullptr);
  EXPECT_FALSE(settings->IsVisible());
}

TEST_F(UIManagerTest, BuildStartScreen_CreatesStartScreenRoot) {
  auto* start_screen = ui_manager_.GetStartScreenRoot();
  ASSERT_NE(start_screen, nullptr);
  EXPECT_EQ(start_screen->GetId(), "start_screen");
  EXPECT_FALSE(start_screen->IsVisible());
}

TEST_F(UIManagerTest, BuildStartScreen_CreatesBeginButton) {
  auto* start_screen = ui_manager_.GetStartScreenRoot();
  ASSERT_NE(start_screen, nullptr);

  auto* btn = start_screen->FindWidget("begin_button");
  ASSERT_NE(btn, nullptr);
}

TEST_F(UIManagerTest, BuildGameOverScreen_CreatesRoot) {
  auto* game_over = ui_manager_.GetGameOverScreenRoot();
  ASSERT_NE(game_over, nullptr);
  EXPECT_EQ(game_over->GetId(), "game_over_screen");
}

TEST_F(UIManagerTest, BuildQuitConfirmMenu_CreatesRoot) {
  auto* quit_confirm = ui_manager_.GetQuitConfirmRoot();
  ASSERT_NE(quit_confirm, nullptr);
  EXPECT_EQ(quit_confirm->GetId(), "quit_confirm_menu");
}

TEST_F(UIManagerTest, BuildChestOpeningScreen_CreatesRoot) {
  auto* chest_opening = ui_manager_.GetChestOpeningRoot();
  ASSERT_NE(chest_opening, nullptr);
  EXPECT_EQ(chest_opening->GetId(), "chest_opening_menu");
}

// =============================================================================
// UIManager Event-Driven Tests (verify event subscriptions trigger UI updates)
// =============================================================================

class UIManagerEventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    resources_ = {};
    ui_manager_.SetupUI(resources_, event_manager_);
    scene_ = testing::CreateTestScene();
  }

  UIResources resources_;
  UIManager ui_manager_;
  EventManager event_manager_;
  Scene scene_;
};

class UIManagerEntityManagerIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    resources_ = {};
    scene_ = testing::CreateTestScene();
    entity_manager_.Initialize(event_manager_);
    ui_manager_.SetupUI(resources_, event_manager_);
  }

  EventContext MakeEventContext() {
    return testing::MakeEventContext(scene_, event_manager_);
  }

  UIResources resources_;
  UIManager ui_manager_;
  EntityManager entity_manager_;
  EventManager event_manager_;
  Scene scene_;
};

TEST_F(UIManagerEventTest, PlayerDamagedEvent_UpdatesHealthBar) {
  EventContext event_context =
      testing::MakeEventContext(scene_, event_manager_);

  scene_.player.stats_.health = 75;
  scene_.player.stats_.max_health.SetBaseValue(100.0f);

  event_manager_.DispatchImmediate(PlayerDamagedEvent{0, 25}, event_context);

  auto* bar = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  ASSERT_NE(bar, nullptr);
  EXPECT_FLOAT_EQ(bar->GetPercent(), 0.75f);

  auto* health_text = ui_manager_.GetWidget<UILabel>("health_text");
  ASSERT_NE(health_text, nullptr);
  EXPECT_EQ(health_text->GetText(), "75/100");
}

TEST_F(UIManagerEventTest, PlayerDamagedEvent_UpdatesHealthBarToZero) {
  EventContext event_context =
      testing::MakeEventContext(scene_, event_manager_);

  scene_.player.stats_.health = 0;
  scene_.player.stats_.max_health.SetBaseValue(100.0f);

  event_manager_.DispatchImmediate(PlayerDamagedEvent{0, 100}, event_context);

  auto* bar = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  ASSERT_NE(bar, nullptr);
  EXPECT_FLOAT_EQ(bar->GetPercent(), 0.0f);
}

TEST_F(UIManagerEventTest, ExpGemCollectedEvent_UpdatesExpBar) {
  EventContext event_context =
      testing::MakeEventContext(scene_, event_manager_);

  scene_.player.stats_.exp_points = 500;
  scene_.player.stats_.exp_points_required.SetBaseValue(1000.0f);

  event_manager_.DispatchImmediate(ExpGemCollectedEvent{0, 200}, event_context);

  auto* exp_bar = ui_manager_.GetWidget<UIProgressBar>("exp_bar");
  ASSERT_NE(exp_bar, nullptr);
  EXPECT_FLOAT_EQ(exp_bar->GetPercent(), 0.5f);

  auto* exp_text = ui_manager_.GetWidget<UILabel>("exp_text");
  ASSERT_NE(exp_text, nullptr);
  EXPECT_EQ(exp_text->GetText(), "500/1000");
}

TEST_F(UIManagerEventTest, PlayerLevelUpEvent_UpdatesExpBarAndLevelText) {
  EventContext event_context =
      testing::MakeEventContext(scene_, event_manager_);

  scene_.player.stats_.exp_points = 0;
  scene_.player.stats_.exp_points_required.SetBaseValue(2000.0f);
  scene_.player.stats_.level = 5;

  event_manager_.DispatchImmediate(PlayerLevelUpEvent{}, event_context);

  auto* exp_text = ui_manager_.GetWidget<UILabel>("exp_text");
  ASSERT_NE(exp_text, nullptr);
  EXPECT_EQ(exp_text->GetText(), "0/2000");

  auto* level_text = ui_manager_.GetWidget<UILabel>("level_text");
  ASSERT_NE(level_text, nullptr);
  EXPECT_EQ(level_text->GetText(), "5");
}

TEST_F(UIManagerEventTest, PlayerClaimedItemEvent_UpdatesInventory) {
  EventContext event_context =
      testing::MakeEventContext(scene_, event_manager_);

  scene_.player.inventory_ = {{ItemId::elia_armor_plate, 2},
                              {ItemId::damodei_claw, 1}};

  event_manager_.DispatchImmediate(PlayerClaimedItemEvent{}, event_context);

  auto* inventory_container =
      ui_manager_.GetRootWidget()->FindWidget("inventory_container");
  ASSERT_NE(inventory_container, nullptr);
  EXPECT_TRUE(inventory_container->IsVisible());

  auto* inventory_bar =
      ui_manager_.GetRootWidget()->FindWidgetAs<HBox>("inventory_bar");
  ASSERT_NE(inventory_bar, nullptr);
  EXPECT_EQ(inventory_bar->GetChildren().size(), 2);

  auto* item_0 =
      inventory_bar->FindWidgetAs<UIInventoryItem>("inventory_item_0");
  ASSERT_NE(item_0, nullptr);
  EXPECT_EQ(item_0->GetItemId(), ItemId::elia_armor_plate);
  EXPECT_EQ(item_0->GetItemCount(), 2);
}

TEST_F(UIManagerEventTest, MultipleEvents_InOrder_UpdateUICorrectly) {
  EventContext event_context =
      testing::MakeEventContext(scene_, event_manager_);

  scene_.player.stats_.health = 100;
  scene_.player.stats_.max_health.SetBaseValue(100.0f);
  scene_.player.stats_.exp_points = 0;
  scene_.player.stats_.exp_points_required.SetBaseValue(1000.0f);

  event_manager_.DispatchImmediate(PlayerDamagedEvent{0, 30}, event_context);
  scene_.player.stats_.health = 70;
  event_manager_.DispatchImmediate(PlayerDamagedEvent{0, 10}, event_context);

  auto* health_bar = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  EXPECT_FLOAT_EQ(health_bar->GetPercent(), 0.7f);

  scene_.player.stats_.exp_points = 150;
  event_manager_.DispatchImmediate(ExpGemCollectedEvent{0, 50}, event_context);

  auto* exp_bar = ui_manager_.GetWidget<UIProgressBar>("exp_bar");
  EXPECT_FLOAT_EQ(exp_bar->GetPercent(), 0.15f);
}

TEST_F(UIManagerEventTest, NoEventHandler_EventDoesNotCrashOrModifyUI) {
  EventContext event_context =
      testing::MakeEventContext(scene_, event_manager_);

  auto root = ui_manager_.GetRootWidget();
  ASSERT_NE(root, nullptr);

  auto* health_bar_before = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  ASSERT_NE(health_bar_before, nullptr);
  float health_percent_before = health_bar_before->GetPercent();

  event_manager_.DispatchImmediate(EnemyKilledEvent{0}, event_context);

  auto* health_bar_after = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  EXPECT_NE(health_bar_after, nullptr);
  EXPECT_FLOAT_EQ(health_bar_after->GetPercent(), health_percent_before);
}

TEST_F(UIManagerEntityManagerIntegrationTest,
       PlayerHealthEvents_UpdateSceneAndHealthBarInSameDispatch) {
  scene_.player.stats_.health = 100;
  scene_.player.stats_.max_health.SetBaseValue(100.0f);
  scene_.player.stats_.armor.SetBaseValue(0.0f);
  scene_.player.is_alive_ = true;
  scene_.player.is_invulnerable = false;

  auto event_context = MakeEventContext();

  event_manager_.DispatchImmediate(PlayerDamagedEvent{0, 25}, event_context);

  EXPECT_EQ(scene_.player.stats_.health, 75);

  auto* health_bar = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  ASSERT_NE(health_bar, nullptr);
  EXPECT_FLOAT_EQ(health_bar->GetPercent(), 0.75f);

  auto* health_text = ui_manager_.GetWidget<UILabel>("health_text");
  ASSERT_NE(health_text, nullptr);
  EXPECT_EQ(health_text->GetText(), "75/100");

  event_manager_.DispatchImmediate(PlayerHealedEvent{10}, event_context);

  EXPECT_EQ(scene_.player.stats_.health, 85);
  EXPECT_FLOAT_EQ(health_bar->GetPercent(), 0.85f);
  EXPECT_EQ(health_text->GetText(), "85/100");
}

}  // namespace
}  // namespace arelto
