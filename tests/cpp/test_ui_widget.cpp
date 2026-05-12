// tests/cpp/test_ui_widget.cpp
// Unit tests for reusable UI widget primitives.

#include <gtest/gtest.h>

#include "items.h"
#include "ui/containers.h"
#include "ui/widget.h"
#include "ui/widgets.h"

namespace arelto {
namespace {

TEST(UIWidgetTest, AddChild_SetsParentAndGrowsChildren) {
  auto parent = std::make_shared<UIWidget>();
  auto child = std::make_shared<UIWidget>();

  parent->AddChild(child);

  EXPECT_EQ(child->GetParent(), parent.get());
  EXPECT_EQ(parent->GetChildren().size(), 1u);
}

TEST(UIWidgetTest, FindWidget_TracesNestedTreeById) {
  auto root = std::make_shared<UIWidget>();
  root->SetId("root");

  auto child = std::make_shared<UIWidget>();
  child->SetId("child");
  auto grandchild = std::make_shared<UIWidget>();
  grandchild->SetId("grandchild");

  root->AddChild(child);
  child->AddChild(grandchild);

  EXPECT_EQ(root->FindWidget("child"), child.get());
  EXPECT_EQ(root->FindWidget("grandchild"), grandchild.get());
  EXPECT_EQ(root->FindWidget("missing"), nullptr);
}

TEST(UIWidgetTest, FindWidgetAs_ReturnsTypedPointerForMatchingWidget) {
  auto root = std::make_shared<UIWidget>();
  auto progress_bar = std::make_shared<UIProgressBar>();
  progress_bar->SetId("progress_bar");
  root->AddChild(progress_bar);

  auto* found = root->FindWidgetAs<UIProgressBar>("progress_bar");
  ASSERT_NE(found, nullptr);

  found->SetPercent(0.5f);
  EXPECT_FLOAT_EQ(found->GetPercent(), 0.5f);
}

TEST(UIWidgetTest, SetVisible_TogglesVisibility) {
  UIWidget widget;
  EXPECT_TRUE(widget.IsVisible());

  widget.SetVisible(false);
  EXPECT_FALSE(widget.IsVisible());

  widget.SetVisible(true);
  EXPECT_TRUE(widget.IsVisible());
}

TEST(VBoxTest, StacksChildrenVerticallyWithSpacing) {
  auto vbox = std::make_shared<VBox>();
  vbox->SetPosition(10, 20);
  vbox->SetSize(200, 400);
  vbox->SetSpacing(5);

  auto child_1 = std::make_shared<UIWidget>();
  child_1->SetSize(100, 30);
  auto child_2 = std::make_shared<UIWidget>();
  child_2->SetSize(100, 40);
  auto child_3 = std::make_shared<UIWidget>();
  child_3->SetSize(100, 50);

  vbox->AddChild(child_1);
  vbox->AddChild(child_2);
  vbox->AddChild(child_3);
  vbox->ComputeLayout(0, 0, 800, 600);

  SDL_Rect first_bounds = child_1->GetComputedBounds();
  SDL_Rect second_bounds = child_2->GetComputedBounds();
  SDL_Rect third_bounds = child_3->GetComputedBounds();

  EXPECT_EQ(first_bounds.y, 20);
  EXPECT_EQ(second_bounds.y, 20 + 30 + 5);
  EXPECT_EQ(third_bounds.y, 20 + 30 + 5 + 40 + 5);
}

TEST(HBoxTest, ArrangesChildrenHorizontallyWithSpacing) {
  auto hbox = std::make_shared<HBox>();
  hbox->SetPosition(10, 20);
  hbox->SetSize(400, 100);
  hbox->SetSpacing(10);

  auto child_1 = std::make_shared<UIWidget>();
  child_1->SetSize(60, 30);
  auto child_2 = std::make_shared<UIWidget>();
  child_2->SetSize(80, 30);

  hbox->AddChild(child_1);
  hbox->AddChild(child_2);
  hbox->ComputeLayout(0, 0, 800, 600);

  SDL_Rect first_bounds = child_1->GetComputedBounds();
  SDL_Rect second_bounds = child_2->GetComputedBounds();

  EXPECT_EQ(first_bounds.x, 10);
  EXPECT_EQ(second_bounds.x, 10 + 60 + 10);
}

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
  EXPECT_EQ(clipped.w, 100);

  bar.SetPercent(1.0f);
  clipped = bar.GetClippedFillSrcRect();
  EXPECT_EQ(clipped.w, 200);
}

TEST(UIProgressBarTest, FillDestRect_UsesComputedBoundsAndFillOffset) {
  auto bar = std::make_shared<UIProgressBar>();
  bar->SetPosition(10, 20);
  bar->SetSize(200, 30);
  bar->SetFillOffset(4, 6);
  bar->SetMaxFillSize(100, 12);
  bar->SetPercent(0.5f);
  bar->ComputeLayout(0, 0, 800, 600);

  SDL_Rect fill_dest = bar->GetFillDestRect();
  EXPECT_EQ(fill_dest.x, 14);
  EXPECT_EQ(fill_dest.y, 26);
  EXPECT_EQ(fill_dest.w, 50);
  EXPECT_EQ(fill_dest.h, 12);
}

TEST(UIButtonTest, HoverState_ChangesCurrentSrcRect) {
  UIButton button;
  button.SetNormalSrcRect({0, 0, 100, 40});
  button.SetHoverSrcRect({0, 40, 100, 40});

  button.SetHovered(false);
  EXPECT_EQ(button.GetCurrentSrcRect().y, 0);

  button.SetHovered(true);
  EXPECT_EQ(button.GetCurrentSrcRect().y, 40);
}

TEST(UICheckboxTest, CheckedStateAndHoverSrcRect_ReflectState) {
  UICheckbox checkbox;
  checkbox.SetBoxSrcRect({0, 0, 30, 30});
  checkbox.SetBoxHoverSrcRect({0, 30, 30, 30});

  EXPECT_FALSE(checkbox.IsChecked());
  checkbox.SetChecked(true);
  EXPECT_TRUE(checkbox.IsChecked());

  checkbox.SetHovered(false);
  EXPECT_EQ(checkbox.GetCurrentBoxSrcRect().y, 0);

  checkbox.SetHovered(true);
  EXPECT_EQ(checkbox.GetCurrentBoxSrcRect().y, 30);
}

TEST(UIAnimationTest, NonLoopingAnimation_StopsOnLastFrame) {
  UIAnimation animation;
  animation.SetFrames({SDL_Rect{0, 0, 10, 10}, SDL_Rect{10, 0, 10, 10}});
  animation.SetFrameDuration(0.1f);
  animation.SetIsLoop(false);
  animation.Play();

  animation.Update(0.25f);

  EXPECT_TRUE(animation.IsFinished());
  EXPECT_EQ(animation.GetCurrentSrcRect().x, 10);
}

TEST(UIInventoryItemTest, StoresItemIdCountAndTexture) {
  UIInventoryItem inventory_item;
  auto* texture = reinterpret_cast<SDL_Texture*>(0x1);

  inventory_item.SetItemId(ItemId::damodei_claw);
  inventory_item.SetItemCount(3);
  inventory_item.SetItemTexture(texture);

  EXPECT_EQ(inventory_item.GetItemId(), ItemId::damodei_claw);
  EXPECT_EQ(inventory_item.GetItemCount(), 3);
  EXPECT_EQ(inventory_item.GetItemTexture(), texture);
}

}  // namespace
}  // namespace arelto
