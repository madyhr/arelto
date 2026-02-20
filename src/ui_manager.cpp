// src/ui_manager.cpp
#include "ui_manager.h"
#include <functional>
#include <string>
#include "constants/progression_manager.h"
#include "constants/projectile.h"
#include "constants/ui.h"
#include "scene.h"

namespace arelto {

void UIManager::SetupUI(const UIResources& resources) {
  resources_ = &resources;

  // The root widget is a full-screen invisible Panel (the "canvas").
  auto root = std::make_shared<Panel>();
  root->SetId("root");
  root->SetSize(kWindowWidth, kWindowHeight);
  root->SetPosition(0, 0);
  root_widget_ = root;

  BuildHUD();
  BuildSettingsMenu();
  BuildStartScreen();
  BuildGameOverScreen();
  BuildQuitConfirmMenu();

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

  auto level_row = std::make_shared<HBox>();
  level_row->SetId("level_row");
  level_row->SetSize(200, kLevelIconSpriteHeight);

  auto level_icon = std::make_shared<UIImage>();
  level_icon->SetId("level_icon");
  level_icon->SetSize(kLevelIconSpriteWidth, kLevelIconSpriteHeight);
  level_icon->SetTexture(resources_->level_indicator_texture);
  level_icon->SetSrcRect({kLevelIconSpriteOffsetX, kLevelIconSpriteOffsetY,
                          kLevelIconSpriteWidth, kLevelIconSpriteHeight});
  level_icon->SetMargin(kLevelUpIconMargin);
  level_row->AddChild(level_icon);

  auto level_text = std::make_shared<UILabel>();
  level_text->SetId("level_text");
  level_text->SetSize(kLevelTextCharWidth * 3, kLevelTextCharHeight);
  level_text->SetUseDigitFont(true);
  level_text->SetDigitSpriteSize(kDigitSpriteWidth, kDigitSpriteHeight);
  level_text->SetCharSize(kLevelTextCharWidth, kLevelTextCharHeight);
  level_row->AddChild(level_text);
  level_text->SetMargin(kLevelUpTextMargin);

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
  // A tranparent overlay to fade out the game while settings menu is present.
  auto overlay = std::make_shared<Panel>();
  overlay->SetId("settings_menu");
  overlay->SetSize(kWindowWidth, kWindowHeight);
  overlay->SetBackgroundColor(WithOpacity(kColorBlack, 128));
  overlay->SetVisible(false);

  // The menu panel itself
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

  auto title_spacer = std::make_shared<Spacer>(0, 5);
  content->AddChild(title_spacer);

  auto title = std::make_shared<UILabel>();
  title->SetId("settings_title");
  title->SetSize(kSettingsMenuWidth - 2 * kMenuContentPadding, 100);
  title->SetText("SETTINGS");
  title->SetFont(resources_->ui_font_huge);
  title->SetCenterWidth(kSettingsMenuWidth - 2 * kMenuContentPadding);
  content->AddChild(title);

  auto volume_label = std::make_shared<UILabel>();
  volume_label->SetId("volume_label");
  volume_label->SetSize(kSettingsMenuWidth - 2 * kMenuContentPadding, 30);
  volume_label->SetText("MUSIC VOLUME");
  volume_label->SetFont(resources_->ui_font_large);
  volume_label->SetCenterWidth(kSettingsMenuWidth - 2 * kMenuContentPadding);
  content->AddChild(volume_label);

  auto volume_slider = std::make_shared<UIProgressBar>();
  volume_slider->SetId("volume_slider");
  volume_slider->SetAnchor(AnchorType::TopCenter);
  volume_slider->SetSize(kSettingsMenuVolumeSliderWidth,
                         kSettingsMenuVolumeSliderHeight);
  volume_slider->SetContainerTexture(resources_->slider_texture);
  volume_slider->SetContainerSrcRect(
      {kSliderContainerSpriteOffsetX, kSliderContainerSpriteOffsetY,
       kSliderContainerSpriteWidth, kSliderContainerSpriteHeight});
  volume_slider->SetFillTexture(resources_->slider_texture);
  volume_slider->SetFillSrcRect({kSliderBarSpriteOffsetX,
                                 kSliderBarSpriteOffsetY, kSliderBarSpriteWidth,
                                 kSliderBarSpriteHeight});
  volume_slider->SetFillOffset(kVolumeSliderFillOffsetX,
                               kVolumeSliderFillOffsetY);
  volume_slider->SetMaxFillSize(kVolumeSliderFillWidth,
                                kVolumeSliderFillHeight);
  volume_slider->SetPercent(1.0f);
  content->AddChild(volume_slider);

  auto mute_row = std::make_shared<HBox>();
  mute_row->SetId("mute_row");
  mute_row->SetSize(kSettingsMenuWidth - 2 * kMenuContentPadding, 40);
  mute_row->SetSpacing(10);
  mute_row->SetAnchor(AnchorType::TopCenter);

  auto mute_checkbox = std::make_shared<UICheckbox>();
  mute_checkbox->SetId("mute_checkbox");
  mute_checkbox->SetSize(30, 30);
  mute_checkbox->SetBoxTexture(resources_->checkbox_texture);
  mute_checkbox->SetBoxSrcRect(
      {0, 0, kCheckboxSpriteWidth, kCheckboxSpriteHeight / 2});
  mute_checkbox->SetBoxHoverSrcRect({0, kCheckboxSpriteHeight / 2,
                                     kCheckboxSpriteWidth,
                                     kCheckboxSpriteHeight / 2});
  mute_checkbox->SetMarkTexture(resources_->checkmark_texture);
  mute_checkbox->SetMarkSrcRect(
      {0, 0, kCheckmarkSpriteWidth, kCheckmarkSpriteHeight});
  mute_row->AddChild(mute_checkbox);

  auto mute_label = std::make_shared<UILabel>();
  mute_label->SetId("mute_label");
  mute_label->SetSize(300, 30);
  mute_label->SetText("Mute Music");
  mute_label->SetFont(resources_->ui_font_medium);
  mute_row->AddChild(mute_label);

  content->AddChild(mute_row);

  auto debug_label = std::make_shared<UILabel>();
  debug_label->SetId("debug_label");
  debug_label->SetSize(kSettingsMenuWidth - 2 * kMenuContentPadding, 40);
  debug_label->SetText("DEBUG");
  debug_label->SetFont(resources_->ui_font_large);
  debug_label->SetCenterWidth(kSettingsMenuWidth - 2 * kMenuContentPadding);
  content->AddChild(debug_label);

  auto occupancy_map_row = std::make_shared<HBox>();
  occupancy_map_row->SetId("occupancy_map_row");
  occupancy_map_row->SetSize(kSettingsMenuWidth - 2 * kMenuContentPadding, 40);
  occupancy_map_row->SetSpacing(10);
  occupancy_map_row->SetAnchor(AnchorType::TopCenter);

  auto occupancy_map_checkbox = std::make_shared<UICheckbox>();
  occupancy_map_checkbox->SetId("occupancy_map_checkbox");
  occupancy_map_checkbox->SetSize(30, 30);
  occupancy_map_checkbox->SetBoxTexture(resources_->checkbox_texture);
  occupancy_map_checkbox->SetBoxSrcRect(
      {0, 0, kCheckboxSpriteWidth, kCheckboxSpriteHeight / 2});
  occupancy_map_checkbox->SetBoxHoverSrcRect({0, kCheckboxSpriteHeight / 2,
                                              kCheckboxSpriteWidth,
                                              kCheckboxSpriteHeight / 2});
  occupancy_map_checkbox->SetMarkTexture(resources_->checkmark_texture);
  occupancy_map_checkbox->SetMarkSrcRect(
      {0, 0, kCheckmarkSpriteWidth, kCheckmarkSpriteHeight});
  occupancy_map_row->AddChild(occupancy_map_checkbox);

  auto occupancy_map_label = std::make_shared<UILabel>();
  occupancy_map_label->SetId("occupancy_map_label");
  occupancy_map_label->SetSize(300, 30);
  occupancy_map_label->SetText("Show Occupancy Map");
  occupancy_map_label->SetFont(resources_->ui_font_medium);
  occupancy_map_row->AddChild(occupancy_map_label);

  content->AddChild(occupancy_map_row);

  auto ray_caster_row = std::make_shared<HBox>();
  ray_caster_row->SetId("ray_caster_row");
  ray_caster_row->SetSize(kSettingsMenuWidth - 2 * kMenuContentPadding, 40);
  ray_caster_row->SetSpacing(10);
  ray_caster_row->SetAnchor(AnchorType::TopCenter);

  auto ray_caster_checkbox = std::make_shared<UICheckbox>();
  ray_caster_checkbox->SetId("ray_caster_checkbox");
  ray_caster_checkbox->SetSize(30, 30);
  ray_caster_checkbox->SetBoxTexture(resources_->checkbox_texture);
  ray_caster_checkbox->SetBoxSrcRect(
      {0, 0, kCheckboxSpriteWidth, kCheckboxSpriteHeight / 2});
  ray_caster_checkbox->SetBoxHoverSrcRect({0, kCheckboxSpriteHeight / 2,
                                           kCheckboxSpriteWidth,
                                           kCheckboxSpriteHeight / 2});
  ray_caster_checkbox->SetMarkTexture(resources_->checkmark_texture);
  ray_caster_checkbox->SetMarkSrcRect(
      {0, 0, kCheckmarkSpriteWidth, kCheckmarkSpriteHeight});
  ray_caster_row->AddChild(ray_caster_checkbox);

  auto ray_caster_label = std::make_shared<UILabel>();
  ray_caster_label->SetId("ray_caster_label");
  ray_caster_label->SetSize(300, 30);
  ray_caster_label->SetText("Show Ray Caster");
  ray_caster_label->SetFont(resources_->ui_font_medium);
  ray_caster_row->AddChild(ray_caster_label);

  content->AddChild(ray_caster_row);

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
      {0, 0, kGenericButtonTextureWidth, kGenericButtonTextureHeight / 2});
  resume_btn->SetHoverSrcRect({0, kGenericButtonTextureHeight / 2,
                               kGenericButtonTextureWidth,
                               kGenericButtonTextureHeight / 2});
  resume_btn->SetLabel("RESUME");
  resume_btn->SetLabelFont(resources_->ui_font_medium);
  button_row->AddChild(resume_btn);

  auto main_menu_btn = std::make_shared<UIButton>();
  main_menu_btn->SetId("main_menu_button");
  main_menu_btn->SetSize(kSettingsMenuButtonWidth, kSettingsMenuButtonHeight);
  main_menu_btn->SetTexture(resources_->button_texture);
  main_menu_btn->SetNormalSrcRect(
      {0, 0, kGenericButtonTextureWidth, kGenericButtonTextureHeight / 2});
  main_menu_btn->SetHoverSrcRect({0, kGenericButtonTextureHeight / 2,
                                  kGenericButtonTextureWidth,
                                  kGenericButtonTextureHeight / 2});
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
  overlay->SetVisible(false);

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

  auto name_label = std::make_shared<UILabel>();
  name_label->SetId(card_id + "_name");
  name_label->SetPosition(kLevelUpNameOffsetX, kLevelUpNameOffsetY);
  name_label->SetSize(kLevelUpCardWidth - 2 * kLevelUpNameOffsetX, 30);
  name_label->SetText(upgrade.GetSpellName());
  name_label->SetFont(resources_->ui_font_large);
  name_label->SetColor({255, 255, 255, 255});
  name_label->SetCenterWidth(kLevelUpCardWidth - 2 * kLevelUpNameOffsetX);
  card->AddChild(name_label);

  auto desc_label = std::make_shared<UILabel>();
  desc_label->SetId(card_id + "_desc");
  desc_label->SetPosition(kLevelUpDescOffsetX, kLevelUpDescOffsetY);
  desc_label->SetSize(kLevelUpCardWidth - 2 * kLevelUpDescOffsetX, 25);
  desc_label->SetText(upgrade.GetDescription());
  desc_label->SetFont(resources_->ui_font_medium);
  desc_label->SetColor({180, 180, 180, 255});
  desc_label->SetCenterWidth(kLevelUpCardWidth - 2 * kLevelUpDescOffsetX);
  card->AddChild(desc_label);

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

  std::string btn_id = "select_button_" + std::to_string(index);
  auto select_btn = std::make_shared<UIButton>();
  select_btn->SetId(btn_id);
  select_btn->SetAnchor(AnchorType::TopCenter);
  select_btn->SetPosition(0, kLevelUpButtonOffsetY);
  select_btn->SetSize(kLevelUpButtonWidth, kLevelUpButtonHeight);
  select_btn->SetTexture(resources_->button_texture);
  select_btn->SetNormalSrcRect(
      {0, 0, kGenericButtonTextureWidth, kGenericButtonTextureHeight / 2});
  select_btn->SetHoverSrcRect({0, kGenericButtonTextureHeight / 2,
                               kGenericButtonTextureWidth,
                               kGenericButtonTextureHeight / 2});
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
// UpdateSettingsMenu — volume slider + mute button text + debug checkboxes
// =============================================================================

void UIManager::UpdateSettingsMenu(float volume, bool is_muted,
                                   const GameStatus& game_status) {
  int mouse_x, mouse_y;
  SDL_GetMouseState(&mouse_x, &mouse_y);

  auto* settings = GetSettingsRoot();

  auto* volume_slider = settings->FindWidgetAs<UIProgressBar>("volume_slider");
  if (volume_slider) {
    // Volume is 0-128.
    float volume_percent = volume / 128.0f;
    volume_slider->SetPercent(volume_percent);
  }

  auto* mute_checkbox = settings->FindWidgetAs<UICheckbox>("mute_checkbox");
  if (mute_checkbox) {
    mute_checkbox->SetChecked(is_muted);
  }

  auto* occupancy_map_checkbox =
      settings->FindWidgetAs<UICheckbox>("occupancy_map_checkbox");
  if (occupancy_map_checkbox) {
    occupancy_map_checkbox->SetChecked(game_status.show_occupancy_map);
  }

  auto* ray_caster_checkbox =
      settings->FindWidgetAs<UICheckbox>("ray_caster_checkbox");
  if (ray_caster_checkbox) {
    ray_caster_checkbox->SetChecked(game_status.show_ray_caster);
  }

  std::function<void(UIWidget*)> update_hover = [&](UIWidget* widget) {
    if (widget->GetWidgetType() == WidgetType::Button ||
        widget->GetWidgetType() == WidgetType::Checkbox) {
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

  auto game_over_image = std::make_shared<UIImage>();
  game_over_image->SetId("game_over_image");
  game_over_image->SetAnchor(AnchorType::Center);
  game_over_image->SetSize(kGameOverSpriteWidth, kGameOverSpriteHeight);
  game_over_image->SetTexture(resources_->game_over_texture);
  game_over_image->SetSrcRect(
      {0, 0, kGameOverSpriteWidth, kGameOverSpriteHeight});

  black_bar->AddChild(game_over_image);

  game_over_root->AddChild(black_bar);
  root_widget_->AddChild(game_over_root);
}

UIWidget* UIManager::GetGameOverScreenRoot() {
  return root_widget_ ? root_widget_->FindWidget("game_over_screen") : nullptr;
}

// =============================================================================
// BuildQuitConfirmMenu
// =============================================================================

void UIManager::BuildQuitConfirmMenu() {
  auto overlay = std::make_shared<Panel>();
  overlay->SetId("quit_confirm_menu");
  overlay->SetSize(kWindowWidth, kWindowHeight);
  overlay->SetBackgroundColor(WithOpacity(kColorBlack, 128));
  overlay->SetVisible(false);

  auto menu = std::make_shared<Panel>();
  menu->SetId("quit_confirm_panel");
  menu->SetAnchor(AnchorType::Center);
  menu->SetSize(kQuitMenuWidth, kQuitMenuHeight);
  menu->SetBackground(resources_->settings_menu_background_texture);
  menu->SetBackgroundSrcRect({0, 0, kSettingsMenuBackgroundSpriteWidth,
                              kSettingsMenuBackgroundSpriteHeight});

  auto content = std::make_shared<VBox>();
  content->SetId("quit_confirm_content");
  content->SetSize(kQuitMenuWidth, kQuitMenuHeight);
  content->SetPadding(kMenuContentPadding);
  content->SetSpacing(kMenuItemSpacing);

  // auto title_spacer = std::make_shared<Spacer>(0, 10);
  // content->AddChild(title_spacer);

  auto title = std::make_shared<UILabel>();
  title->SetId("quit_confirm_title");
  title->SetSize(kQuitMenuWidth - 2 * kMenuContentPadding, 100);
  title->SetText("QUIT GAME?");
  title->SetFont(resources_->ui_font_huge);
  title->SetCenterWidth(kQuitMenuWidth - 2 * kMenuContentPadding);
  content->AddChild(title);

  menu->AddChild(content);

  auto button_row = std::make_shared<HBox>();
  button_row->SetId("quit_confirm_buttons");
  button_row->SetAnchor(AnchorType::BottomCenter);
  button_row->SetPosition(0, -kMenuBottomPadding);
  button_row->SetSize(kSettingsMenuButtonWidth * 2 + kMenuButtonGap,
                      kSettingsMenuButtonHeight);
  button_row->SetSpacing(kMenuButtonGap);

  auto yes_btn = std::make_shared<UIButton>();
  yes_btn->SetId("quit_yes_button");
  yes_btn->SetSize(kSettingsMenuButtonWidth, kSettingsMenuButtonHeight);
  yes_btn->SetTexture(resources_->button_texture);
  yes_btn->SetNormalSrcRect(
      {0, 0, kGenericButtonTextureWidth, kGenericButtonTextureHeight / 2});
  yes_btn->SetHoverSrcRect({0, kGenericButtonTextureHeight / 2,
                            kGenericButtonTextureWidth,
                            kGenericButtonTextureHeight / 2});
  yes_btn->SetLabel("YES");
  yes_btn->SetLabelFont(resources_->ui_font_medium);
  button_row->AddChild(yes_btn);

  auto no_btn = std::make_shared<UIButton>();
  no_btn->SetId("quit_no_button");
  no_btn->SetSize(kSettingsMenuButtonWidth, kSettingsMenuButtonHeight);
  no_btn->SetTexture(resources_->button_texture);
  no_btn->SetNormalSrcRect(
      {0, 0, kGenericButtonTextureWidth, kGenericButtonTextureHeight / 2});
  no_btn->SetHoverSrcRect({0, kGenericButtonTextureHeight / 2,
                           kGenericButtonTextureWidth,
                           kGenericButtonTextureHeight / 2});
  no_btn->SetLabel("NO");
  no_btn->SetLabelFont(resources_->ui_font_medium);
  button_row->AddChild(no_btn);

  menu->AddChild(button_row);

  overlay->AddChild(menu);
  root_widget_->AddChild(overlay);
}

UIWidget* UIManager::GetQuitConfirmRoot() {
  return root_widget_ ? root_widget_->FindWidget("quit_confirm_menu") : nullptr;
}

void UIManager::UpdateQuitConfirmMenu() {
  auto* quit_confirm = GetQuitConfirmRoot();
  if (!quit_confirm)
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
  update_hover(quit_confirm);
}

}  // namespace arelto
