#ifndef RL2_CONFIG_UI_CONFIG_H_
#define RL2_CONFIG_UI_CONFIG_H_

#include <SDL2/SDL_pixels.h>
#include "constants/game.h"

namespace arelto {

struct UIColorsConfig {
  SDL_Color positive_green = SDL_Color{76, 201, 118, 255};
  SDL_Color negative_red = SDL_Color{235, 87, 87, 255};
};

struct UIFontConfig {
  int kFontSizeSmall = 14;
  int kFontSizeMedium = 26;
  int kFontSizeLarge = 40;
  int kFontSizeHuge = 72;
};

struct UIHudConfig {
  int kHudPadding = 50;
  int kHudBarSpacing = 8;
  int kBarTextOffsetX = 50;
  int kBarTextOffsetY = 0;
  int kLevelGroupOffsetY = 55;
  int kTimerTextGap = 10;
  int kLevelTextGap = -2;

  int kDigitSpriteWidth = 30;
  int kDigitSpriteHeight = 50;

  int kHealthBarContainerSpriteOffsetX = 0;
  int kHealthBarContainerSpriteOffsetY = 0;
  int kHealthBarContainerSpriteWidth = 404;
  int kHealthBarContainerSpriteHeight = 92;
  float kHealthBarRelOffsetX = 80.0f;
  float kHealthBarRelOffsetY = 32.0f;
  int kHealthBarSpriteOffsetX = 0;
  int kHealthBarSpriteOffsetY = 128;
  int kHealthBarSpriteWidth = 299;
  int kHealthBarSpriteHeight = 28;
  int kHealthBarTextRelOffsetX = 100;
  int kHealthBarTextRelOffsetY = 32;
  int kHealthBarTextCharWidth = 20;
  int kHealthBarTextCharHeight = 25;

  int kTimerHourglassSpriteWidth = 50;
  int kTimerHourglassSpriteHeight = 72;
  int kTimerTextCharWidth = 50;
  int kTimerTextCharHeight = 72;

  int kGameOverSpriteWidth = 610;
  int kGameOverSpriteHeight = 88;

  int kExpBarContainerSpriteOffsetX = 0;
  int kExpBarContainerSpriteOffsetY = 0;
  int kExpBarContainerSpriteWidth = 404;
  int kExpBarContainerSpriteHeight = 92;
  float kExpBarRelOffsetX = 80.0f;
  float kExpBarRelOffsetY = 30.0f;
  int kExpBarSpriteOffsetX = 0;
  int kExpBarSpriteOffsetY = 128;
  int kExpBarSpriteWidth = 299;
  int kExpBarSpriteHeight = 28;
  int kExpBarTextRelOffsetX = 100;
  int kExpBarTextRelOffsetY = 32;
  int kExpBarTextCharWidth = 20;
  int kExpBarTextCharHeight = 25;

  int kLevelIconSpriteOffsetX = 0;
  int kLevelIconSpriteOffsetY = 0;
  int kLevelIconSpriteWidth = 70;
  int kLevelIconSpriteHeight = 74;
  int kLevelTextCharWidth = 50;
  int kLevelTextCharHeight = 72;
  int kLevelUpIconMargin = -10;
  int kLevelUpTextMargin = -10;
};

struct UIMenuConfig {
  int kMenuContentPadding = 100;
  int kMenuItemSpacing = 25;
  int kMenuButtonGap = 20;
  int kMenuBottomPadding = 60;

  int kGenericButtonTextureWidth = 300;
  int kGenericButtonTextureHeight = 160;

  int kBeginButtonTextureWidth = 638;
  int kBeginButtonTextureHeight = 540;
  int kBeginButtonWidth = 450;
  int kBeginButtonHeight = 175;
  int kBeginButtonY = 2 * (kWindowHeight - kBeginButtonHeight) / 7;

  int kSettingsMenuWidth = 450;
  int kSettingsMenuHeight = 750;
  int kSettingsMenuBackgroundSpriteWidth = 900;
  int kSettingsMenuBackgroundSpriteHeight = 1000;
  int kSettingsMenuButtonWidth = 150;
  int kSettingsMenuButtonHeight = 50;
  int kSettingsMenuVolumeSliderWidth = 300;
  int kSettingsMenuVolumeSliderHeight = 30;
  int kVolumeSliderFillOffsetX = 15;
  int kVolumeSliderFillOffsetY = 5;
  int kVolumeSliderFillWidth = 275;
  int kVolumeSliderFillHeight = 20;

  int kQuitMenuWidth = 550;
  int kQuitMenuHeight = 300;

  int kSliderContainerSpriteOffsetX = 0;
  int kSliderContainerSpriteOffsetY = 0;
  int kSliderContainerSpriteWidth = 882;
  int kSliderContainerSpriteHeight = 48;
  int kSliderBarSpriteOffsetX = 0;
  int kSliderBarSpriteOffsetY = 48;
  int kSliderBarSpriteWidth = 806;
  int kSliderBarSpriteHeight = 29;

  int kCheckboxSpriteWidth = 263;
  int kCheckboxSpriteHeight = 526;
  int kCheckmarkSpriteWidth = 193;
  int kCheckmarkSpriteHeight = 164;
};

struct UICardConfig {
  int kLevelUpCardWidth = 400;
  int kLevelUpCardHeight = 600;
  int kLevelUpCardGap = 100;
  int kLevelUpIconOffsetY = 120;
  int kLevelUpIconSize = 80;
  int kLevelUpNameOffsetY = 220;
  int kLevelUpNameOffsetX = 70;
  int kLevelUpDescOffsetY = 300;
  int kLevelUpDescOffsetX = 70;
  int kLevelUpStatsOffsetY = 350;
  int kLevelUpStatsOffsetX = 70;
  int kLevelUpRowStride = 55;
  int kLevelUpButtonOffsetY = 440;
  int kLevelUpButtonWidth = 200;
  int kLevelUpButtonHeight = 50;

  int kItemIconSize = 300;
  int kItemCardWidth = 650;
  int kItemCardHeight = 1000;
  int kItemCardGap = 150;
  int kItemCardIconOffsetY = 160;
  int kItemCardIconSize = 80;
  int kItemCardNameOffsetY = 475;
  int kItemCardNameOffsetX = 100;
  int kItemCardDescOffsetY = 600;
  int kItemCardDescOffsetX = 70;
  int kItemCardStatsOffsetY = 648;
  int kItemCardStatsOffsetX = 70;
  int kItemCardRowStride = 100;
  int kItemCardButtonOffsetY = 800;
  int kItemCardButtonWidth = 200;
  int kItemCardButtonHeight = 50;
};

struct UIInventoryConfig {
  int kInventoryBarY = 50;
  int kInventoryIconSize = 60;
  int kInventoryWidgetHeight = 60;
  int kInventoryLabelWidth = 20;
  int kInventoryItemGap = 15;
  int kInventoryMultiplierSize = 16;
  int kInventoryMultiplierMargin = 0;
  int kInventoryContainerPadding = 20;
  int kInventoryBackgroundAlpha = 64;
};

struct UIConfig {
  UIColorsConfig colors;
  UIFontConfig fonts;
  UIHudConfig hud;
  UIMenuConfig menus;
  UICardConfig cards;
  UIInventoryConfig inventory;
};

UIConfig MakeDefaultUIConfig();

}  // namespace arelto

#endif
