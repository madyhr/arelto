// src/ui_manager.cpp
#include "ui_manager.h"
#include <algorithm>
#include <string>
#include "constants/ui.h"
#include "scene.h"

namespace arelto {

void UIManager::SetupUI(const UIResources& resources) {
  resources_ = &resources;

  // The root widget is a full-screen invisible Panel (the "Canvas").
  auto root = std::make_shared<Panel>();
  root->SetId("root");
  root->SetSize(kWindowWidth, kWindowHeight);
  root->SetPosition(0, 0);
  root_widget_ = root;

  BuildHUD();
  BuildSettingsMenu();

  // Compute initial layout from the screen origin.
  root_widget_->ComputeLayout(0, 0, kWindowWidth, kWindowHeight);
}

UIWidget* UIManager::GetRootWidget() {
  return root_widget_.get();
}

UIWidget* UIManager::GetSettingsRoot() {
  return root_widget_ ? root_widget_->FindWidget("settings_menu") : nullptr;
}

// =============================================================================
// BuildHUD — Health Bar, Exp Bar, Level Indicator, Timer
// =============================================================================

void UIManager::BuildHUD() {
  // --- Top-left group: Timer + Level Indicator ---
  auto top_left = std::make_shared<VBox>();
  top_left->SetId("hud_top_left");
  top_left->SetAnchor(AnchorType::TopLeft);
  top_left->SetPosition(kHudPadding, kHudPadding);
  top_left->SetSize(200, 200);
  top_left->SetSpacing(kLevelGroupOffsetY);

  // Timer row (hourglass + text)
  auto timer_row = std::make_shared<HBox>();
  timer_row->SetId("timer_row");
  timer_row->SetSize(200, kTimerHourglassSpriteHeight);
  timer_row->SetSpacing(kTimerTextGap);

  auto timer_icon = std::make_shared<UIImage>();
  timer_icon->SetId("timer_icon");
  timer_icon->SetSize(kTimerHourglassSpriteWidth, kTimerHourglassSpriteHeight);
  timer_icon->SetTexture(resources_->timer_hourglass_texture);
  timer_icon->SetSrcRect(
      {0, 0, kTimerHourglassSpriteWidth, kTimerHourglassSpriteHeight});
  timer_row->AddChild(timer_icon);

  auto timer_text = std::make_shared<UILabel>();
  timer_text->SetId("timer_text");
  timer_text->SetSize(kTimerTextCharWidth * 3, kTimerTextCharHeight);
  timer_text->SetUseDigitFont(true);
  timer_text->SetDigitSpriteSize(kDigitSpriteWidth, kDigitSpriteHeight);
  timer_text->SetCharSize(kTimerTextCharWidth, kTimerTextCharHeight);
  timer_row->AddChild(timer_text);

  top_left->AddChild(timer_row);

  // Level row (icon + text)
  auto level_row = std::make_shared<HBox>();
  level_row->SetId("level_row");
  level_row->SetSize(200, kLevelIconSpriteHeight);
  level_row->SetSpacing(kLevelTextGap);

  auto level_icon = std::make_shared<UIImage>();
  level_icon->SetId("level_icon");
  level_icon->SetSize(kLevelIconSpriteWidth, kLevelIconSpriteHeight);
  level_icon->SetTexture(resources_->level_indicator_texture);
  level_icon->SetSrcRect({kLevelIconSpriteOffsetX, kLevelIconSpriteOffsetY,
                          kLevelIconSpriteWidth, kLevelIconSpriteHeight});
  level_row->AddChild(level_icon);

  auto level_text = std::make_shared<UILabel>();
  level_text->SetId("level_text");
  level_text->SetSize(kLevelTextCharWidth * 3, kLevelTextCharHeight);
  level_text->SetUseDigitFont(true);
  level_text->SetDigitSpriteSize(kDigitSpriteWidth, kDigitSpriteHeight);
  level_text->SetCharSize(kLevelTextCharWidth, kLevelTextCharHeight);
  level_row->AddChild(level_text);

  top_left->AddChild(level_row);
  root_widget_->AddChild(top_left);

  // --- Bottom-left group: Health Bar + Exp Bar ---
  auto bottom_left = std::make_shared<VBox>();
  bottom_left->SetId("hud_bottom_left");
  bottom_left->SetAnchor(AnchorType::BottomLeft);
  bottom_left->SetPosition(kHudPadding, -kHudPadding);
  bottom_left->SetSize(kHealthBarContainerSpriteWidth,
                       kHealthBarContainerSpriteHeight * 2 + kHudBarSpacing);
  bottom_left->SetSpacing(kHudBarSpacing);

  // Health bar (with overlaid text)
  auto health_bar = std::make_shared<UIProgressBar>();
  health_bar->SetId("health_bar");
  health_bar->SetSize(kHealthBarContainerSpriteWidth,
                      kHealthBarContainerSpriteHeight);
  health_bar->SetContainerTexture(resources_->health_bar_texture);
  health_bar->SetContainerSrcRect(
      {kHealthBarContainerSpriteOffsetX, kHealthBarContainerSpriteOffsetY,
       kHealthBarContainerSpriteWidth, kHealthBarContainerSpriteHeight});
  health_bar->SetFillTexture(resources_->health_bar_texture);
  health_bar->SetFillSrcRect({kHealthBarSpriteOffsetX, kHealthBarSpriteOffsetY,
                              kHealthBarSpriteWidth, kHealthBarSpriteHeight});
  health_bar->SetFillOffset(kHealthBarRelOffsetX, kHealthBarRelOffsetY);
  health_bar->SetMaxFillSize(kHealthBarSpriteWidth, kHealthBarSpriteHeight);
  health_bar->SetPercent(1.0f);

  // Health text overlay — child of health_bar so it's positioned relative to bar
  auto health_text = std::make_shared<UILabel>();
  health_text->SetId("health_text");
  health_text->SetPosition(kHealthBarTextRelOffsetX, kHealthBarTextRelOffsetY);
  health_text->SetSize(kDigitSpriteWidth, kDigitSpriteHeight);
  health_text->SetUseDigitFont(true);
  health_text->SetDigitSpriteSize(kDigitSpriteWidth, kDigitSpriteHeight);
  health_text->SetCharSize(kHealthBarTextCharWidth, kHealthBarTextCharHeight);
  health_bar->AddChild(health_text);

  bottom_left->AddChild(health_bar);

  // Exp bar
  auto exp_bar = std::make_shared<UIProgressBar>();
  exp_bar->SetId("exp_bar");
  exp_bar->SetSize(kExpBarContainerSpriteWidth, kExpBarContainerSpriteHeight);
  exp_bar->SetContainerTexture(resources_->exp_bar_texture);
  exp_bar->SetContainerSrcRect(
      {kExpBarContainerSpriteOffsetX, kExpBarContainerSpriteOffsetY,
       kExpBarContainerSpriteWidth, kExpBarContainerSpriteHeight});
  exp_bar->SetFillTexture(resources_->exp_bar_texture);
  exp_bar->SetFillSrcRect({kExpBarSpriteOffsetX, kExpBarSpriteOffsetY,
                           kExpBarSpriteWidth, kExpBarSpriteHeight});
  exp_bar->SetFillOffset(kExpBarRelOffsetX, kExpBarRelOffsetY);
  exp_bar->SetMaxFillSize(kExpBarSpriteWidth, kExpBarSpriteHeight);
  exp_bar->SetPercent(0.0f);

  // Exp text overlay — child of exp_bar
  auto exp_text = std::make_shared<UILabel>();
  exp_text->SetId("exp_text");
  exp_text->SetPosition(kExpBarTextRelOffsetX, kExpBarTextRelOffsetY);
  exp_text->SetSize(kDigitSpriteWidth, kDigitSpriteHeight);
  exp_text->SetUseDigitFont(true);
  exp_text->SetDigitSpriteSize(kDigitSpriteWidth, kDigitSpriteHeight);
  exp_text->SetCharSize(kExpBarTextCharWidth, kExpBarTextCharHeight);
  exp_bar->AddChild(exp_text);

  bottom_left->AddChild(exp_bar);

  root_widget_->AddChild(bottom_left);
}

// =============================================================================
// BuildSettingsMenu
// =============================================================================

void UIManager::BuildSettingsMenu() {
  // Settings panel — centered on screen
  auto menu = std::make_shared<Panel>();
  menu->SetId("settings_menu");
  menu->SetAnchor(AnchorType::Center);
  menu->SetSize(kSettingsMenuWidth, kSettingsMenuHeight);
  menu->SetBackground(resources_->settings_menu_background_texture);
  menu->SetBackgroundSrcRect({0, 0, kSettingsMenuBackgroundSpriteWidth,
                              kSettingsMenuBackgroundSpriteHeight});
  menu->SetVisible(false);

  // Content laid out as a VBox inside the menu
  auto content = std::make_shared<VBox>();
  content->SetId("settings_content");
  content->SetSize(kSettingsMenuWidth, kSettingsMenuHeight);
  content->SetPadding(kMenuContentPadding);
  content->SetSpacing(kMenuItemSpacing);

  // Title: centered at top
  auto title = std::make_shared<UILabel>();
  title->SetId("settings_title");
  title->SetSize(kSettingsMenuWidth - 2 * kMenuContentPadding, 50);
  title->SetText("SETTINGS");
  title->SetFont(resources_->ui_font_huge);
  title->SetCenterWidth(kSettingsMenuWidth - 2 * kMenuContentPadding);
  content->AddChild(title);

  // Volume Label
  auto vol_label = std::make_shared<UILabel>();
  vol_label->SetId("volume_label");
  vol_label->SetSize(kSettingsMenuWidth - 2 * kMenuContentPadding, 30);
  vol_label->SetText("MUSIC VOLUME");
  vol_label->SetFont(resources_->ui_font_large);
  vol_label->SetCenterWidth(kSettingsMenuWidth - 2 * kMenuContentPadding);
  content->AddChild(vol_label);

  // Volume Slider: centered horizontally
  auto vol_slider = std::make_shared<UIProgressBar>();
  vol_slider->SetId("volume_slider");
  vol_slider->SetAnchor(AnchorType::TopCenter);
  vol_slider->SetSize(kSettingsMenuVolumeSliderWidth,
                      kSettingsMenuVolumeSliderHeight);
  vol_slider->SetContainerTexture(resources_->slider_texture);
  vol_slider->SetContainerSrcRect(
      {kSliderContainerSpriteOffsetX, kSliderContainerSpriteOffsetY,
       kSliderContainerSpriteWidth, kSliderContainerSpriteHeight});
  vol_slider->SetFillTexture(resources_->slider_texture);
  vol_slider->SetFillSrcRect({kSliderBarSpriteOffsetX, kSliderBarSpriteOffsetY,
                              kSliderBarSpriteWidth, kSliderBarSpriteHeight});
  vol_slider->SetFillOffset(kVolumeSliderFillOffsetX, kVolumeSliderFillOffsetY);
  vol_slider->SetMaxFillSize(kVolumeSliderFillWidth, kVolumeSliderFillHeight);
  vol_slider->SetPercent(1.0f);
  content->AddChild(vol_slider);

  // Mute Button: centered horizontally
  auto mute_btn = std::make_shared<UIButton>();
  mute_btn->SetId("mute_button");
  mute_btn->SetAnchor(AnchorType::TopCenter);
  mute_btn->SetSize(kSettingsMenuButtonWidth, kSettingsMenuButtonHeight);
  mute_btn->SetTexture(resources_->button_texture);
  mute_btn->SetNormalSrcRect(
      {0, 0, kLevelUpButtonTextureWidth, kLevelUpButtonTextureHeight / 2});
  mute_btn->SetHoverSrcRect({0, kLevelUpButtonTextureHeight / 2,
                             kLevelUpButtonTextureWidth,
                             kLevelUpButtonTextureHeight / 2});
  mute_btn->SetLabel("MUTE");
  mute_btn->SetLabelFont(resources_->ui_font_medium);
  content->AddChild(mute_btn);

  menu->AddChild(content);

  // Bottom button row: Resume + Main Menu, anchored to bottom
  auto button_row = std::make_shared<HBox>();
  button_row->SetId("settings_buttons");
  button_row->SetAnchor(AnchorType::BottomCenter);
  button_row->SetPosition(0, -kMenuBottomPadding);
  button_row->SetSize(kSettingsMenuButtonWidth * 2 + kMenuButtonGap,
                      kSettingsMenuButtonHeight);
  button_row->SetSpacing(kMenuButtonGap);

  auto resume_btn = std::make_shared<UIButton>();
  resume_btn->SetId("resume_button");
  resume_btn->SetSize(kSettingsMenuButtonWidth, kSettingsMenuButtonHeight);
  resume_btn->SetTexture(resources_->button_texture);
  resume_btn->SetNormalSrcRect(
      {0, 0, kLevelUpButtonTextureWidth, kLevelUpButtonTextureHeight / 2});
  resume_btn->SetHoverSrcRect({0, kLevelUpButtonTextureHeight / 2,
                               kLevelUpButtonTextureWidth,
                               kLevelUpButtonTextureHeight / 2});
  resume_btn->SetLabel("RESUME");
  resume_btn->SetLabelFont(resources_->ui_font_medium);
  button_row->AddChild(resume_btn);

  auto main_menu_btn = std::make_shared<UIButton>();
  main_menu_btn->SetId("main_menu_button");
  main_menu_btn->SetSize(kSettingsMenuButtonWidth, kSettingsMenuButtonHeight);
  main_menu_btn->SetTexture(resources_->button_texture);
  main_menu_btn->SetNormalSrcRect(
      {0, 0, kLevelUpButtonTextureWidth, kLevelUpButtonTextureHeight / 2});
  main_menu_btn->SetHoverSrcRect({0, kLevelUpButtonTextureHeight / 2,
                                  kLevelUpButtonTextureWidth,
                                  kLevelUpButtonTextureHeight / 2});
  main_menu_btn->SetLabel("MAIN MENU");
  main_menu_btn->SetLabelFont(resources_->ui_font_medium);
  button_row->AddChild(main_menu_btn);

  menu->AddChild(button_row);
  root_widget_->AddChild(menu);
}

// =============================================================================
// Update — refresh dynamic widget state from Scene data
// =============================================================================

void UIManager::Update(const Scene& scene, float time) {
  // Health bar
  auto* health_bar = GetWidget<UIProgressBar>("health_bar");
  if (health_bar) {
    int current_hp = scene.player.stats_.health;
    int max_hp = scene.player.stats_.max_health;
    float percent = static_cast<float>(current_hp) / max_hp;
    health_bar->SetPercent(percent);
  }

  auto* health_text = GetWidget<UILabel>("health_text");
  if (health_text) {
    health_text->SetText(std::to_string(scene.player.stats_.health) + "/" +
                         std::to_string(scene.player.stats_.max_health));
  }

  // Exp bar
  auto* exp_bar = GetWidget<UIProgressBar>("exp_bar");
  if (exp_bar) {
    int current_exp = scene.player.stats_.exp_points;
    int max_exp = scene.player.stats_.exp_points_required;
    float percent = static_cast<float>(current_exp) / max_exp;
    exp_bar->SetPercent(percent);
  }

  auto* exp_text = GetWidget<UILabel>("exp_text");
  if (exp_text) {
    exp_text->SetText(std::to_string(scene.player.stats_.exp_points) + "/" +
                      std::to_string(scene.player.stats_.exp_points_required));
  }

  // Level indicator
  auto* level_text = GetWidget<UILabel>("level_text");
  if (level_text) {
    level_text->SetText(std::to_string(scene.player.stats_.level));
  }

  // Timer
  auto* timer_text = GetWidget<UILabel>("timer_text");
  if (timer_text) {
    timer_text->SetText(std::to_string(static_cast<int>(time)));
  }
}

// =============================================================================
// UpdateSettingsMenu — volume slider + mute button text
// =============================================================================

void UIManager::UpdateSettingsMenu(float volume, bool is_muted) {
  auto* vol_slider = GetWidget<UIProgressBar>("volume_slider");
  if (vol_slider) {
    float vol_percent = std::clamp(volume / 128.0f, 0.0f, 1.0f);
    vol_slider->SetPercent(vol_percent);
  }

  auto* mute_btn = GetWidget<UIButton>("mute_button");
  if (mute_btn) {
    mute_btn->SetLabel(is_muted ? "UNMUTE" : "MUTE");
  }

  // Update hover state for all buttons in the settings menu
  int mouse_x, mouse_y;
  SDL_GetMouseState(&mouse_x, &mouse_y);

  auto* settings = GetSettingsRoot();
  if (!settings)
    return;

  // Recursively check buttons (they're now nested in containers)
  std::function<void(UIWidget*)> update_hover = [&](UIWidget* widget) {
    if (widget->GetWidgetType() == WidgetType::Button) {
      SDL_Rect bounds = widget->GetComputedBounds();
      bool hovered = (mouse_x >= bounds.x && mouse_x <= bounds.x + bounds.w &&
                      mouse_y >= bounds.y && mouse_y <= bounds.y + bounds.h);
      widget->SetHovered(hovered);
    }
    for (auto& child : widget->GetChildren()) {
      update_hover(child.get());
    }
  };
  update_hover(settings);
}

}  // namespace arelto
