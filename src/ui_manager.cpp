// src/ui_manager.cpp
#include "ui_manager.h"
#include <algorithm>
#include <string>
#include "constants/ui.h"
#include "scene.h"
#include "types.h"

namespace rl2 {

void UIManager::SetupUI() {
  SetupHealthBar();
  SetupExpBar();
  SetupLevelIndicator();
  SetupTimer();
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

}  // namespace rl2
