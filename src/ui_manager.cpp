// src/ui_manager.cpp
#include "ui_manager.h"
#include "constants.h"
#include "scene.h"

namespace rl2 {

void UIManager::SetupUI() {
  SetupHealthBar();
  SetupTimer();
};

void UIManager::SetupHealthBar() {
  health_bar_.type = UIElementGroupType::health_bar;
  health_bar_.screen_position = {kHealthBarGroupX, kHealthBarGroupY};

  UIElement health_bar_container = {
      {kHealthBarContainerSpriteOffsetX, kHealthBarContainerSpriteOffsetY,
       kHealthBarContainerSpriteWidth, kHealthBarContainerSpriteHeight},
      {kHealthBarContainerRelOffsetX, kHealthBarContainerRelOffsetY},
      {kHealthBarContainerSpriteWidth, kHealthBarContainerSpriteHeight},
      UIElement::Tag::background};

  UIElement health_bar_fill = {
      {kHealthBarSpriteOffsetX, kHealthBarSpriteOffsetY, kHealthBarSpriteWidth,
       kHealthBarSpriteHeight},
      {kHealthBarRelOffsetX, kHealthBarRelOffsetY},
      {kHealthBarSpriteWidth, kHealthBarSpriteHeight},
      UIElement::Tag::fill};

  // Order matters here, as the first element in elements is also rendered first.
  health_bar_.elements.push_back(health_bar_container);
  health_bar_.elements.push_back(health_bar_fill);
}

void UIManager::SetupTimer() {
  timer_.type = UIElementGroupType::timer;
  timer_.screen_position = {kTimerGroupX, kTimerGroupY};

  // the texture map only contains the hourglass, so sprite offset is {0,0}.
  UIElement timer_icon = {
      {0, 0, kTimerHourglassSpriteWidth, kTimerHourglassSpriteHeight},
      {kTimerHourglassRelOffsetX, kTimerHourglassRelOffsetY},
      {kTimerHourglassSpriteWidth, kTimerHourglassSpriteHeight},
      UIElement::Tag::icon};

  timer_.elements.push_back(timer_icon);
};

void UIManager::UpdateUI(const Scene& scene) {
  UpdateHealthBar(scene.player.stats_.health, scene.player.stats_.max_health);
};

void UIManager::UpdateHealthBar(int current_hp, int max_hp) {
  float percent = static_cast<float>(current_hp) / max_hp;

  if (percent < 0.0f)
    percent = 0.0f;
  if (percent > 1.0f)
    percent = 1.0f;

  UIElement* fill_bar = health_bar_.GetElemByTag(UIElement::Tag::fill);

  if (fill_bar) {
    fill_bar->sprite_size.width =
        static_cast<int>(kHealthBarSpriteWidth * percent);
    fill_bar->src_rect.w = static_cast<int>(kHealthBarSpriteWidth * percent);
  }
};

}  // namespace rl2
