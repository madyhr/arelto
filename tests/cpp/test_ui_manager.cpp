// tests/cpp/test_ui_manager.cpp
// Unit tests for UIManager with widget tree

#include <gtest/gtest.h>

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

TEST(UIWidgetTest, SetVisible_Works) {
  UIWidget w;
  w.SetVisible(false);
  EXPECT_FALSE(w.IsVisible());
  w.SetVisible(true);
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
// UILabel Tests
// =============================================================================

TEST(UILabelTest, SetText_UpdatesValue) {
  UILabel lbl;
  lbl.SetText("100/200");
  EXPECT_EQ(lbl.GetText(), "100/200");
  lbl.SetText("50/200");
  EXPECT_EQ(lbl.GetText(), "50/200");
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
// UIManager Integration Tests (no SDL needed — just tree construction)
// =============================================================================

class UIManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // UIResources with nullptr textures/fonts — we just verify tree structure
    resources_ = {};
    ui_manager_.SetupUI(resources_);
    scene_ = testing::CreateTestScene();
  }

  UIResources resources_;
  UIManager ui_manager_;
  Scene scene_;
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

TEST_F(UIManagerTest, SettingsMenu_StartsHidden) {
  auto* settings = ui_manager_.GetSettingsRoot();
  ASSERT_NE(settings, nullptr);
  EXPECT_FALSE(settings->IsVisible());
}

TEST_F(UIManagerTest, Update_ChangesHealthBarPercent) {
  scene_.player.stats_.health = 50;
  scene_.player.stats_.max_health = 100;
  ui_manager_.Update(scene_, 0.0f);

  auto* bar = ui_manager_.GetWidget<UIProgressBar>("health_bar");
  ASSERT_NE(bar, nullptr);
  EXPECT_FLOAT_EQ(bar->GetPercent(), 0.5f);
}

TEST_F(UIManagerTest, Update_ChangesTimerText) {
  ui_manager_.Update(scene_, 65.0f);
  auto* label = ui_manager_.GetWidget<UILabel>("timer_text");
  ASSERT_NE(label, nullptr);
  EXPECT_EQ(label->GetText(), "65");
}

TEST_F(UIManagerTest, Update_ChangesLevelText) {
  scene_.player.stats_.level = 5;
  ui_manager_.Update(scene_, 0.0f);
  auto* label = ui_manager_.GetWidget<UILabel>("level_text");
  ASSERT_NE(label, nullptr);
  EXPECT_EQ(label->GetText(), "5");
}

TEST_F(UIManagerTest, BuildStartScreen_CreatesStartScreenRoot) {
  ui_manager_.BuildStartScreen();
  auto* start_screen = ui_manager_.GetStartScreenRoot();
  ASSERT_NE(start_screen, nullptr);
  EXPECT_EQ(start_screen->GetId(), "start_screen");
  EXPECT_FALSE(start_screen->IsVisible());
}

TEST_F(UIManagerTest, BuildStartScreen_CreatesBeginButton) {
  ui_manager_.BuildStartScreen();
  auto* start_screen = ui_manager_.GetStartScreenRoot();
  ASSERT_NE(start_screen, nullptr);

  auto* btn = start_screen->FindWidget("begin_button");
  ASSERT_NE(btn, nullptr);
}

}  // namespace
}  // namespace arelto
