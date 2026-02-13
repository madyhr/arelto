// src/ui_manager.cpp
#include "ui_manager.h"
#include <algorithm>
#include <string>
#include "constants/ui.h"
#include "scene.h"
#include "types.h"

namespace arelto {

void UIManager::SetupUI() {
  SetupHealthBar();
  SetupExpBar();
  SetupLevelIndicator();
  SetupTimer();
  SetupSettingsMenu();
};

void UIManager::SetupHealthBar() {
  health_bar_.type = UIElementGroupType::health_bar;
  health_bar_.screen_position = {kHealthBarGroupX, kHealthBarGroupY};

  UIElement health_bar_container = {
      SDL_Rect{kHealthBarContainerSpriteOffsetX,
               kHealthBarContainerSpriteOffsetY, kHealthBarContainerSpriteWidth,
               kHealthBarContainerSpriteHeight},
      Vector2D{kHealthBarContainerRelOffsetX, kHealthBarContainerRelOffsetY},
      Size2D{kHealthBarContainerSpriteWidth, kHealthBarContainerSpriteHeight},
      UIElement::Tag::background};

  UIElement health_bar_fill = {
      SDL_Rect{kHealthBarSpriteOffsetX, kHealthBarSpriteOffsetY,
               kHealthBarSpriteWidth, kHealthBarSpriteHeight},
      Vector2D{kHealthBarRelOffsetX, kHealthBarRelOffsetY},
      Size2D{kHealthBarSpriteWidth, kHealthBarSpriteHeight},
      UIElement::Tag::fill};

  UIElement health_bar_text = {
      SDL_Rect{0, 0, kDigitSpriteWidth, kDigitSpriteHeight},
      Vector2D{kHealthBarTextRelOffsetX, kHealthBarTextRelOffsetY},
      Size2D{kDigitSpriteWidth, kDigitSpriteHeight}, UIElement::Tag::text,
      Size2D{kHealthBarTextCharWidth, kHealthBarTextCharHeight}};

  // Order matters here, as the first element in elements is also rendered first.
  health_bar_.elements.push_back(health_bar_container);
  health_bar_.elements.push_back(health_bar_fill);
  health_bar_.elements.push_back(health_bar_text);
}

void UIManager::SetupExpBar() {
  exp_bar_.type = UIElementGroupType::exp_bar;
  exp_bar_.screen_position = {kExpBarGroupX, kExpBarGroupY};

  UIElement exp_bar_container = {
      SDL_Rect{kExpBarContainerSpriteOffsetX, kExpBarContainerSpriteOffsetY,
               kExpBarContainerSpriteWidth, kExpBarContainerSpriteHeight},
      Vector2D{kExpBarContainerRelOffsetX, kExpBarContainerRelOffsetY},
      Size2D{kExpBarContainerSpriteWidth, kExpBarContainerSpriteHeight},
      UIElement::Tag::background};

  UIElement exp_bar_fill = {SDL_Rect{kExpBarSpriteOffsetX, kExpBarSpriteOffsetY,
                                     kExpBarSpriteWidth, kExpBarSpriteHeight},
                            Vector2D{kExpBarRelOffsetX, kExpBarRelOffsetY},
                            Size2D{kExpBarSpriteWidth, kExpBarSpriteHeight},
                            UIElement::Tag::fill};

  UIElement exp_bar_text = {
      SDL_Rect{0, 0, kDigitSpriteWidth, kDigitSpriteHeight},
      Vector2D{kExpBarTextRelOffsetX, kExpBarTextRelOffsetY},
      Size2D{kDigitSpriteWidth, kDigitSpriteHeight}, UIElement::Tag::text,
      Size2D{kExpBarTextCharWidth, kExpBarTextCharHeight}};

  // Order matters here, as the first element in elements is also rendered first.
  exp_bar_.elements.push_back(exp_bar_container);
  exp_bar_.elements.push_back(exp_bar_fill);
  exp_bar_.elements.push_back(exp_bar_text);
}

void UIManager::SetupLevelIndicator() {
  level_indicator_.type = UIElementGroupType::level_indicator;
  level_indicator_.screen_position = {kLevelGroupX, kLevelGroupY};

  UIElement level_indicator_icon = {
      SDL_Rect{kLevelIconSpriteOffsetX, kLevelIconSpriteOffsetY,
               kLevelIconSpriteWidth, kLevelIconSpriteHeight},
      Vector2D{kLevelIconRelOffsetX, kLevelIconRelOffsetY},
      Size2D{kLevelIconSpriteWidth, kLevelIconSpriteHeight},
      UIElement::Tag::icon};

  UIElement level_indicator_text = {
      SDL_Rect{0, 0, kDigitSpriteWidth, kDigitSpriteHeight},
      Vector2D{kLevelTextRelOffsetX, kLevelTextRelOffsetY},
      Size2D{kDigitSpriteWidth, kDigitSpriteHeight}, UIElement::Tag::text,
      Size2D{kLevelTextCharWidth, kLevelTextCharHeight}};

  // Order matters here, as the first element in elements is also rendered first.
  level_indicator_.elements.push_back(level_indicator_icon);
  level_indicator_.elements.push_back(level_indicator_text);
}

void UIManager::SetupTimer() {
  timer_.type = UIElementGroupType::timer;
  timer_.screen_position = {kTimerGroupX, kTimerGroupY};

  // the texture map only contains the hourglass, so sprite offset is {0,0}.
  UIElement timer_icon = {
      SDL_Rect{0, 0, kTimerHourglassSpriteWidth, kTimerHourglassSpriteHeight},
      Vector2D{kTimerHourglassRelOffsetX, kTimerHourglassRelOffsetY},
      Size2D{kTimerHourglassSpriteWidth, kTimerHourglassSpriteHeight},
      UIElement::Tag::icon};

  UIElement timer_text = {SDL_Rect{0, 0, kDigitSpriteWidth, kDigitSpriteHeight},
                          Vector2D{kTimerTextRelOffsetX, kTimerTextRelOffsetY},
                          Size2D{kDigitSpriteWidth, kDigitSpriteHeight},
                          UIElement::Tag::text,
                          Size2D{kTimerTextCharWidth, kTimerTextCharHeight}};

  timer_.elements.push_back(timer_icon);
  timer_.elements.push_back(timer_text);
};

void UIManager::SetupSettingsBackground() {
  UIElement title_text = {
      SDL_Rect{0, 0, kDigitSpriteWidth, kDigitSpriteHeight},
      Vector2D{0.0f, static_cast<float>(kSettingsMenuTitleY)},
      Size2D{kDigitSpriteWidth, kDigitSpriteHeight}, UIElement::Tag::text,
      Size2D{30, 40}};
  title_text.text_value = "SETTINGS";

  UIElement settings_menu_background = {
      SDL_Rect{0, 0, kSettingsMenuBackgroundSpriteWidth,
               kSettingsMenuBackgroundSpriteHeight},
      {0.0, 0.0},
      {kSettingsMenuWidth, kSettingsMenuHeight},
      UIElement::Tag::background};

  settings_menu_.module_elements.push_back(settings_menu_background);
  settings_menu_.module_elements.push_back(title_text);
}

void UIManager::SetupSettingsVolumeControl() {
  UIElementGroup volume_control;
  volume_control.group_tag = UIElementGroup::GroupTag::volume_control;
  volume_control.screen_position =
      settings_menu_.screen_position +
      Vector2D{kSettingsMenuVolumeControlGroupRelX,
               kSettingsMenuVolumeControlGroupRelY};

  UIElement volume_label = {
      SDL_Rect{0, 0, 0, 0},
      Vector2D{0.0f, static_cast<float>(kSettingsMenuVolumeY)}, Size2D{0, 0},
      UIElement::Tag::text, Size2D{20, 25}};
  volume_label.text_value = "MUSIC VOLUME";
  volume_control.elements.push_back(volume_label);

  UIElement volume_slider_background = {
      SDL_Rect{kSliderContainerSpriteOffsetX, kSliderContainerSpriteOffsetY,
               kSliderContainerSpriteWidth, kSliderContainerSpriteHeight},
      Vector2D{static_cast<float>(kSettingsMenuVolumeSliderX),
               static_cast<float>(kSettingsMenuVolumeSliderY)},
      Size2D{kSettingsMenuVolumeSliderWidth, kSettingsMenuVolumeSliderHeight},
      UIElement::Tag::background};
  volume_control.elements.push_back(volume_slider_background);

  UIElement volume_slider_fill = {
      SDL_Rect{kSliderBarSpriteOffsetX, kSliderBarSpriteOffsetY,
               kSliderBarSpriteWidth, kSliderBarSpriteHeight},
      Vector2D{static_cast<float>(kSettingsMenuVolumeSliderX +
                                  kVolumeSliderFillOffsetX),
               static_cast<float>(kSettingsMenuVolumeSliderY +
                                  kVolumeSliderFillOffsetY)},
      Size2D{kVolumeSliderFillWidth, kVolumeSliderFillHeight},
      UIElement::Tag::fill};
  volume_control.elements.push_back(volume_slider_fill);

  UIElement mute_btn = {
      SDL_Rect{0, 0, kLevelUpButtonTextureWidth, kLevelUpButtonTextureHeight},
      Vector2D{static_cast<float>(kSettingsMenuButtonX),
               static_cast<float>(kSettingsMenuMuteY)},
      Size2D{kSettingsMenuButtonWidth, kSettingsMenuButtonHeight},
      UIElement::Tag::button};
  mute_btn.text_value = "MUTE";
  volume_control.elements.push_back(mute_btn);

  settings_menu_.element_groups.push_back(volume_control);
}

void UIManager::SetupSettingsMainMenu() {
  UIElementGroup main_menu_settings;
  main_menu_settings.group_tag = UIElementGroup::GroupTag::main_menu_settings;
  main_menu_settings.screen_position = settings_menu_.screen_position;

  UIElement main_menu_btn = {
      SDL_Rect{0, 0, kLevelUpButtonTextureWidth, kLevelUpButtonTextureHeight},
      Vector2D{static_cast<float>(kSettingsMenuMainMenuX),
               static_cast<float>(kSettingsMenuMainMenuY)},
      Size2D{kSettingsMenuButtonWidth, kSettingsMenuButtonHeight},
      UIElement::Tag::button};
  main_menu_btn.text_value = "MAIN MENU";
  main_menu_settings.elements.push_back(main_menu_btn);

  settings_menu_.element_groups.push_back(main_menu_settings);
}

void UIManager::SetupSettingsResume() {
  UIElementGroup resume_settings;
  resume_settings.group_tag = UIElementGroup::GroupTag::resume_settings;
  resume_settings.screen_position = settings_menu_.screen_position;

  UIElement resume_btn = {
      SDL_Rect{0, 0, kLevelUpButtonTextureWidth, kLevelUpButtonTextureHeight},
      Vector2D{static_cast<float>(kSettingsMenuResumeX),
               static_cast<float>(kSettingsMenuResumeY)},
      Size2D{kSettingsMenuButtonWidth, kSettingsMenuButtonHeight},
      UIElement::Tag::button};
  resume_btn.text_value = "RESUME";
  resume_settings.elements.push_back(resume_btn);

  settings_menu_.element_groups.push_back(resume_settings);
}

void UIManager::SetupSettingsMenu() {
  settings_menu_.type = UIElementGroupType::settings_menu;
  settings_menu_.screen_position = {static_cast<float>(kSettingsMenuX),
                                    static_cast<float>(kSettingsMenuY)};

  SetupSettingsBackground();
  SetupSettingsVolumeControl();
  SetupSettingsMainMenu();
  SetupSettingsResume();
}

void UIManager::UpdateUI(const Scene& scene, float time) {
  UpdateHealthBar(scene);
  UpdateExpBar(scene);
  UpdateLevelIndicator(scene);
  UpdateTimer(time);
};

void UIManager::UpdateHealthBar(const Scene& scene) {
  int current_hp = scene.player.stats_.health;
  int max_hp = scene.player.stats_.max_health;
  float percent = static_cast<float>(current_hp) / max_hp;
  // the percentage is clamped to avoid having a negative sprite width;
  percent = std::clamp(percent, 0.0f, 1.0f);

  UIElement* fill_bar = health_bar_.GetElemByTag(UIElement::Tag::fill);

  if (fill_bar) {
    fill_bar->sprite_size.width =
        static_cast<int>(kHealthBarSpriteWidth * percent);
    fill_bar->src_rect.w = static_cast<int>(kHealthBarSpriteWidth * percent);
  }

  UIElement* text_elem = health_bar_.GetElemByTag(UIElement::Tag::text);

  if (text_elem) {
    text_elem->text_value =
        std::to_string(current_hp) + "/" + std::to_string(max_hp);
  }
};

void UIManager::UpdateExpBar(const Scene& scene) {
  int current_exp = scene.player.stats_.exp_points;
  int max_exp = scene.player.stats_.exp_points_required;
  float percent = static_cast<float>(current_exp) / max_exp;
  // the percentage is clamped to avoid having a negative sprite width;
  percent = std::clamp(percent, 0.0f, 1.0f);

  UIElement* fill_bar = exp_bar_.GetElemByTag(UIElement::Tag::fill);

  if (fill_bar) {
    fill_bar->sprite_size.width =
        static_cast<int>(kExpBarSpriteWidth * percent);
    fill_bar->src_rect.w = static_cast<int>(kExpBarSpriteWidth * percent);
  }

  UIElement* text_elem = exp_bar_.GetElemByTag(UIElement::Tag::text);

  if (text_elem) {
    text_elem->text_value =
        std::to_string(current_exp) + "/" + std::to_string(max_exp);
  }
};

void UIManager::UpdateLevelIndicator(const Scene& scene) {
  int level = scene.player.stats_.level;
  UIElement* text_elem = level_indicator_.GetElemByTag(UIElement::Tag::text);

  if (text_elem) {
    text_elem->text_value = std::to_string(level);
  }
};

void UIManager::UpdateTimer(float time) {
  UIElement* text_elem = timer_.GetElemByTag(UIElement::Tag::text);
  if (text_elem) {
    text_elem->text_value = std::to_string(static_cast<int>(time));
  }
};

void UIManager::UpdateSettingsMenu(float volume, bool is_muted) {
  // Update volume slider fill
  UIElementGroup* volume_control = settings_menu_.GetElemGroupByTag(
      UIElementGroup::GroupTag::volume_control);

  if (!volume_control) {
    return;
  }

  UIElement* fill = volume_control->GetElemByTag(UIElement::Tag::fill);
  if (fill) {
    float vol_percent = std::clamp(volume / 128.0f, 0.0f, 1.0f);
    fill->sprite_size.width =
        static_cast<int>(kVolumeSliderFillWidth * vol_percent);
    fill->src_rect.w = static_cast<int>(kSliderBarSpriteWidth * vol_percent);
  }

  // Update mute text
  UIElement* mute_txt = volume_control->GetElemByTag(UIElement::Tag::button);
  if (mute_txt) {
    mute_txt->text_value = is_muted ? "UNMUTE" : "MUTE";
  }
}

}  // namespace arelto
