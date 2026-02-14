// src/ui_manager.cpp
#include "ui_manager.h"
#include <algorithm>
#include <functional>
#include <string>
#include "constants/progression_manager.h"
#include "constants/projectile.h"
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
  BuildStartScreen();
  BuildGameOverScreen();

  // Compute initial layout from the screen origin.
  root_widget_->ComputeLayout(0, 0, kWindowWidth, kWindowHeight);
}

UIWidget* UIManager::GetRootWidget() {
  return root_widget_.get();
}

UIWidget* UIManager::GetSettingsRoot() {
  return root_widget_ ? root_widget_->FindWidget("settings_menu") : nullptr;
}

UIWidget* UIManager::GetLevelUpRoot() {
  return root_widget_ ? root_widget_->FindWidget("level_up_menu") : nullptr;
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
  // Fullscreen wrapper for the overlay
  auto overlay = std::make_shared<Panel>();
  overlay->SetId("settings_menu");  // Keep ID same for root lookup
  overlay->SetSize(kWindowWidth, kWindowHeight);
  overlay->SetBackgroundColor(WithOpacity(kColorBlack, 128));
  overlay->SetVisible(false);

  // The actual menu panel
  auto menu = std::make_shared<Panel>();
  menu->SetId("settings_panel");
  menu->SetAnchor(AnchorType::Center);
  menu->SetSize(kSettingsMenuWidth, kSettingsMenuHeight);
  menu->SetBackground(resources_->settings_menu_background_texture);
  menu->SetBackgroundSrcRect({0, 0, kSettingsMenuBackgroundSpriteWidth,
                              kSettingsMenuBackgroundSpriteHeight});

  auto content = std::make_shared<VBox>();
  content->SetId("settings_content");
  content->SetSize(kSettingsMenuWidth, kSettingsMenuHeight);
  content->SetPadding(kMenuContentPadding);
  content->SetSpacing(kMenuItemSpacing);

  auto title = std::make_shared<UILabel>();
  title->SetId("settings_title");
  title->SetSize(kSettingsMenuWidth - 2 * kMenuContentPadding, 50);
  title->SetText("SETTINGS");
  title->SetFont(resources_->ui_font_huge);
  title->SetCenterWidth(kSettingsMenuWidth - 2 * kMenuContentPadding);
  content->AddChild(title);

  auto vol_label = std::make_shared<UILabel>();
  vol_label->SetId("volume_label");
  vol_label->SetSize(kSettingsMenuWidth - 2 * kMenuContentPadding, 30);
  vol_label->SetText("MUSIC VOLUME");
  vol_label->SetFont(resources_->ui_font_large);
  vol_label->SetCenterWidth(kSettingsMenuWidth - 2 * kMenuContentPadding);
  content->AddChild(vol_label);

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

  overlay->AddChild(menu);
  root_widget_->AddChild(overlay);
}

// =============================================================================
// BuildLevelUpMenu — dynamically create card widgets from upgrade options
// =============================================================================

void UIManager::BuildLevelUpMenu(
    const std::vector<std::unique_ptr<Upgrade>>& options) {
  root_widget_->RemoveChild("level_up_menu");

  auto overlay = std::make_shared<Panel>();
  overlay->SetId("level_up_menu");
  overlay->SetSize(kWindowWidth, kWindowHeight);
  overlay->SetBackgroundColor(WithOpacity(kColorBlack, 128));
  overlay->SetVisible(true);

  auto card_row = std::make_shared<HBox>();
  card_row->SetId("level_up_cards");
  card_row->SetAnchor(AnchorType::Center);
  int total_width = kNumUpgradeOptions * kLevelUpCardWidth +
                    (kNumUpgradeOptions - 1) * kLevelUpCardGap;
  card_row->SetSize(total_width, kLevelUpCardHeight);
  card_row->SetSpacing(kLevelUpCardGap);

  for (size_t i = 0; i < options.size(); ++i) {
    BuildLevelUpCard(card_row.get(), static_cast<int>(i), *options[i]);
  }

  overlay->AddChild(card_row);
  root_widget_->AddChild(overlay);
  root_widget_->ComputeLayout(0, 0, kWindowWidth, kWindowHeight);
}

void UIManager::BuildLevelUpCard(UIWidget* parent, int index,
                                 const Upgrade& upgrade) {
  std::string card_id = "level_up_card_" + std::to_string(index);

  auto card = std::make_shared<Panel>();
  card->SetId(card_id);
  card->SetSize(kLevelUpCardWidth, kLevelUpCardHeight);
  card->SetBackground(resources_->level_up_option_card_texture);
  card->SetBackgroundSrcRect({0, 0, 0, 0});  // full texture

  // Spell icon — centered horizontally
  int spell_id = upgrade.GetSpellID();
  if (spell_id >= 0 &&
      spell_id < static_cast<int>(resources_->projectile_textures.size())) {
    auto icon = std::make_shared<UIImage>();
    icon->SetId(card_id + "_icon");
    icon->SetAnchor(AnchorType::TopCenter);
    icon->SetPosition(0, kLevelUpIconOffsetY);
    icon->SetSize(kLevelUpIconSize, kLevelUpIconSize);
    icon->SetTexture(resources_->projectile_textures[spell_id]);
    icon->SetSrcRect({0, 0, kFireballSpriteWidth, kFireballSpriteHeight});
    card->AddChild(icon);
  }

  // Spell name
  auto name_label = std::make_shared<UILabel>();
  name_label->SetId(card_id + "_name");
  name_label->SetPosition(kLevelUpNameOffsetX, kLevelUpNameOffsetY);
  name_label->SetSize(kLevelUpCardWidth - 2 * kLevelUpNameOffsetX, 30);
  name_label->SetText(upgrade.GetSpellName());
  name_label->SetFont(resources_->ui_font_large);
  name_label->SetColor({255, 255, 255, 255});
  name_label->SetCenterWidth(kLevelUpCardWidth - 2 * kLevelUpNameOffsetX);
  card->AddChild(name_label);

  // Description
  auto desc_label = std::make_shared<UILabel>();
  desc_label->SetId(card_id + "_desc");
  desc_label->SetPosition(kLevelUpDescOffsetX, kLevelUpDescOffsetY);
  desc_label->SetSize(kLevelUpCardWidth - 2 * kLevelUpDescOffsetX, 25);
  desc_label->SetText(upgrade.GetDescription());
  desc_label->SetFont(resources_->ui_font_medium);
  desc_label->SetColor({180, 180, 180, 255});
  desc_label->SetCenterWidth(kLevelUpCardWidth - 2 * kLevelUpDescOffsetX);
  card->AddChild(desc_label);

  // Value change (e.g. "1.00 -> 0.90")
  std::string stats_str =
      upgrade.GetOldValueString() + " -> " + upgrade.GetNewValueString();
  auto stats_label = std::make_shared<UILabel>();
  stats_label->SetId(card_id + "_stats");
  stats_label->SetPosition(kLevelUpStatsOffsetX, kLevelUpStatsOffsetY);
  stats_label->SetSize(kLevelUpCardWidth - 2 * kLevelUpStatsOffsetX, 25);
  stats_label->SetText(stats_str);
  stats_label->SetFont(resources_->ui_font_medium);
  stats_label->SetColor({0, 255, 0, 255});
  stats_label->SetCenterWidth(kLevelUpCardWidth - 2 * kLevelUpStatsOffsetX);
  card->AddChild(stats_label);

  // SELECT button
  std::string btn_id = "select_button_" + std::to_string(index);
  auto select_btn = std::make_shared<UIButton>();
  select_btn->SetId(btn_id);
  select_btn->SetAnchor(AnchorType::TopCenter);
  select_btn->SetPosition(0, kLevelUpButtonOffsetY);
  select_btn->SetSize(kLevelUpButtonWidth, kLevelUpButtonHeight);
  select_btn->SetTexture(resources_->button_texture);
  select_btn->SetNormalSrcRect(
      {0, 0, kLevelUpButtonTextureWidth, kLevelUpButtonTextureHeight / 2});
  select_btn->SetHoverSrcRect({0, kLevelUpButtonTextureHeight / 2,
                               kLevelUpButtonTextureWidth,
                               kLevelUpButtonTextureHeight / 2});
  select_btn->SetLabel("SELECT");
  select_btn->SetLabelFont(resources_->ui_font_medium);
  card->AddChild(select_btn);

  parent->AddChild(card);
}

void UIManager::UpdateLevelUpMenu() {
  auto* level_up = GetLevelUpRoot();
  if (!level_up)
    return;

  int mouse_x, mouse_y;
  SDL_GetMouseState(&mouse_x, &mouse_y);

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
  update_hover(level_up);
}

// =============================================================================
// Update — refresh dynamic widget state from Scene data
// =============================================================================

void UIManager::Update(const Scene& scene, float time) {
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

  auto* level_text = GetWidget<UILabel>("level_text");
  if (level_text) {
    level_text->SetText(std::to_string(scene.player.stats_.level));
  }

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

  int mouse_x, mouse_y;
  SDL_GetMouseState(&mouse_x, &mouse_y);

  auto* settings = GetSettingsRoot();
  if (!settings)
    return;

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

// =============================================================================
// BuildStartScreen
// =============================================================================

void UIManager::BuildStartScreen() {
  auto start_screen = std::make_shared<Panel>();
  start_screen->SetId("start_screen");
  start_screen->SetSize(kWindowWidth, kWindowHeight);
  start_screen->SetBackground(resources_->start_screen_texture);
  start_screen->SetBackgroundSrcRect({0, 0, 0, 0});  // full texture
  start_screen->SetVisible(false);

  auto begin_btn = std::make_shared<UIButton>();
  begin_btn->SetId("begin_button");
  begin_btn->SetAnchor(AnchorType::BottomCenter);
  // Original Y = 5/7 * (H - h).
  // Distance from bottom = H - (Y + h) = 2/7 * (H - h).
  // SetPosition uses positive Y for down, so negative Y for up from bottom.
  int offset_y = -2 * (kWindowHeight - kBeginButtonHeight) / 7;
  begin_btn->SetPosition(0, offset_y);
  begin_btn->SetSize(kBeginButtonWidth, kBeginButtonHeight);
  begin_btn->SetTexture(resources_->begin_button_texture);
  begin_btn->SetNormalSrcRect(
      {0, 0, kBeginButtonTextureWidth, kBeginButtonTextureHeight / 2});
  begin_btn->SetHoverSrcRect({0, kBeginButtonTextureHeight / 2,
                              kBeginButtonTextureWidth,
                              kBeginButtonTextureHeight / 2});

  start_screen->AddChild(begin_btn);
  root_widget_->AddChild(start_screen);
}

UIWidget* UIManager::GetStartScreenRoot() {
  return root_widget_ ? root_widget_->FindWidget("start_screen") : nullptr;
}

void UIManager::UpdateStartScreen() {
  auto* start_screen = GetStartScreenRoot();
  if (!start_screen)
    return;

  int mouse_x, mouse_y;
  SDL_GetMouseState(&mouse_x, &mouse_y);

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
  update_hover(start_screen);
}

// =============================================================================
// BuildGameOverScreen
// =============================================================================

void UIManager::BuildGameOverScreen() {
  auto game_over_root = std::make_shared<Panel>();
  game_over_root->SetId("game_over_screen");
  game_over_root->SetSize(kWindowWidth, kWindowHeight);
  game_over_root->SetVisible(false);

  auto black_bar = std::make_shared<Panel>();
  black_bar->SetId("game_over_bar");
  black_bar->SetAnchor(AnchorType::Center);
  black_bar->SetSize(kWindowWidth, kWindowHeight / 3);
  black_bar->SetBackgroundColor(WithOpacity(kColorBlack, 128));

  auto go_image = std::make_shared<UIImage>();
  go_image->SetId("game_over_image");
  go_image->SetAnchor(AnchorType::Center);
  go_image->SetSize(kGameOverSpriteWidth, kGameOverSpriteHeight);
  go_image->SetTexture(resources_->game_over_texture);
  go_image->SetSrcRect({0, 0, kGameOverSpriteWidth, kGameOverSpriteHeight});

  black_bar->AddChild(go_image);

  game_over_root->AddChild(black_bar);
  root_widget_->AddChild(game_over_root);
}

UIWidget* UIManager::GetGameOverScreenRoot() {
  return root_widget_ ? root_widget_->FindWidget("game_over_screen") : nullptr;
}

}  // namespace arelto
