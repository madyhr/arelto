// include/constants/ui.h
#ifndef RL2_CONSTANTS_UI_H_
#define RL2_CONSTANTS_UI_H_

#include "constants/game.h"
namespace arelto {

// Layout constants — internal padding / spacing for widget tree.
constexpr int kHudPadding = 50;  // Distance from screen edges
constexpr int kHudBarSpacing = 8;
constexpr int kBarTextOffsetX = 50;
constexpr int kBarTextOffsetY = 0;
constexpr int kLevelGroupOffsetY = 55;
constexpr int kTimerTextGap = 10;
constexpr int kLevelTextGap = -2;  // Set to a negative value due to icon size

// Settings menu internal layout
constexpr int kMenuContentPadding = 100;
constexpr int kMenuItemSpacing = 25;
constexpr int kMenuButtonGap = 20;
constexpr int kMenuBottomPadding = 60;

// Font sizes
constexpr int kFontSizeMedium = 26;
constexpr int kFontSizeLarge = 40;
constexpr int kFontSizeHuge = 72;

// Digits
constexpr int kDigitSpriteWidth = 30;
constexpr int kDigitSpriteHeight = 50;

// Health bar
constexpr int kHealthBarContainerSpriteOffsetX = 0;
constexpr int kHealthBarContainerSpriteOffsetY = 0;
constexpr int kHealthBarContainerSpriteWidth = 404;
constexpr int kHealthBarContainerSpriteHeight = 92;
constexpr float kHealthBarRelOffsetX = 80.0f;
constexpr float kHealthBarRelOffsetY = 32.0f;
constexpr int kHealthBarSpriteOffsetX = 0;
constexpr int kHealthBarSpriteOffsetY = 128;
constexpr int kHealthBarSpriteWidth = 299;
constexpr int kHealthBarSpriteHeight = 28;
constexpr int kHealthBarTextRelOffsetX = 100;
constexpr int kHealthBarTextRelOffsetY = 32;
constexpr int kHealthBarTextCharWidth = 20;
constexpr int kHealthBarTextCharHeight = 25;

// Timer
constexpr int kTimerHourglassSpriteWidth = 50;
constexpr int kTimerHourglassSpriteHeight = 72;
constexpr int kTimerTextCharWidth = 50;
constexpr int kTimerTextCharHeight = 72;

// Game Over text
constexpr int kGameOverSpriteWidth = 610;
constexpr int kGameOverSpriteHeight = 88;

// Experience Bar
constexpr int kExpBarContainerSpriteOffsetX = 0;
constexpr int kExpBarContainerSpriteOffsetY = 0;
constexpr int kExpBarContainerSpriteWidth = 404;
constexpr int kExpBarContainerSpriteHeight = 92;
constexpr float kExpBarRelOffsetX = 80.0f;
constexpr float kExpBarRelOffsetY = 30.0f;
constexpr int kExpBarSpriteOffsetX = 0;
constexpr int kExpBarSpriteOffsetY = 128;
constexpr int kExpBarSpriteWidth = 299;
constexpr int kExpBarSpriteHeight = 28;
constexpr int kExpBarTextRelOffsetX = 100;
constexpr int kExpBarTextRelOffsetY = 32;
constexpr int kExpBarTextCharWidth = 20;
constexpr int kExpBarTextCharHeight = 25;

// Level indicator
constexpr int kLevelIconSpriteOffsetX = 0;
constexpr int kLevelIconSpriteOffsetY = 0;
constexpr int kLevelIconSpriteWidth = 70;
constexpr int kLevelIconSpriteHeight = 74;
constexpr int kLevelTextCharWidth = 50;
constexpr int kLevelTextCharHeight = 72;
constexpr int kLevelUpIconMargin = -10;
constexpr int kLevelUpTextMargin = -10;

// Level Up Option Card
constexpr int kLevelUpCardWidth = 400;
constexpr int kLevelUpCardHeight = 600;
constexpr int kLevelUpCardGap = 100;
// offsets relative to card top-left
constexpr int kLevelUpIconOffsetY = 120;
constexpr int kLevelUpIconSize = 80;
constexpr int kLevelUpNameOffsetY = 220;
constexpr int kLevelUpNameOffsetX = 70;
constexpr int kLevelUpDescOffsetY = 300;
constexpr int kLevelUpDescOffsetX = 70;
constexpr int kLevelUpStatsOffsetY = 350;
constexpr int kLevelUpStatsOffsetX = 70;

// Level Up Button
constexpr int kLevelUpButtonTextureWidth = 300;
constexpr int kLevelUpButtonTextureHeight = 160;
constexpr int kLevelUpButtonWidth = 200;
constexpr int kLevelUpButtonHeight = 50;
constexpr int kLevelUpButtonOffsetY = 440;  // from card top

// Start screen
constexpr int kBeginButtonTextureWidth = 638;
constexpr int kBeginButtonTextureHeight = 540;
constexpr int kBeginButtonWidth = 450;
constexpr int kBeginButtonHeight = 175;
constexpr int kBeginButtonX = (kWindowWidth - kBeginButtonWidth) / 2;
constexpr int kBeginButtonY = 5 * (kWindowHeight - kBeginButtonHeight) / 7;

constexpr int kSettingsMenuWidth = 450;
constexpr int kSettingsMenuHeight = 750;
constexpr int kSettingsMenuBackgroundSpriteWidth = 900;
constexpr int kSettingsMenuBackgroundSpriteHeight = 1000;

constexpr int kSettingsMenuButtonWidth = 150;
constexpr int kSettingsMenuButtonHeight = 50;
constexpr int kSettingsMenuVolumeSliderWidth = 300;
constexpr int kSettingsMenuVolumeSliderHeight = 30;

constexpr int kVolumeSliderFillOffsetX = 15;
constexpr int kVolumeSliderFillOffsetY = 5;
constexpr int kVolumeSliderFillWidth = 275;
constexpr int kVolumeSliderFillHeight = 20;

// Generic slider and fill texture
constexpr int kSliderContainerSpriteOffsetX = 0;
constexpr int kSliderContainerSpriteOffsetY = 0;
constexpr int kSliderContainerSpriteWidth = 882;
constexpr int kSliderContainerSpriteHeight = 48;
constexpr int kSliderBarSpriteOffsetX = 0;
constexpr int kSliderBarSpriteOffsetY = 48;
constexpr int kSliderBarSpriteWidth = 806;
constexpr int kSliderBarSpriteHeight = 29;

// Generic checkbox and checkmark texture
constexpr int kCheckboxSpriteWidth = 263;
constexpr int kCheckboxSpriteHeight = 526;
constexpr int kCheckmarkSpriteWidth = 193;
constexpr int kCheckmarkSpriteHeight = 164;

}  // namespace arelto

#endif
