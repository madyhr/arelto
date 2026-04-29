#ifndef RL2_CONFIG_UI_CONFIG_YAML_H_
#define RL2_CONFIG_UI_CONFIG_YAML_H_

#include <iostream>
#include "config/ui_config.h"
#include "ui_config.h"
#include "yaml-cpp/yaml.h"

namespace arelto::config::detail {

inline void DecodeColorChannel(const YAML::Node& node, const char* key,
                               Uint8& channel, const char* owner) {
  const YAML::Node field = node[key];
  if (!field) {
    return;
  }

  try {
    const int parsed = field.as<int>();
    if (parsed < 0 || parsed > 255) {
      std::cerr << "Invalid UI config value for " << owner << "." << key
                << ": expected [0, 255], got " << parsed << '\n';
      return;
    }
    channel = static_cast<Uint8>(parsed);
  } catch (const YAML::Exception& e) {
    std::cerr << "Invalid UI config value for " << owner << "." << key << ": "
              << e.what() << '\n';
  }
}

template <typename T>
void DecodeField(const YAML::Node& node, const char* key, T& out,
                 const char* owner) {
  const YAML::Node field = node[key];
  if (!field) {
    return;
  }

  try {
    out = field.as<T>();
  } catch (const YAML::Exception& e) {
    std::cerr << "Invalid UI config value for " << owner << "." << key << ": "
              << e.what() << '\n';
  }
}

inline void DecodeField(const YAML::Node& node, const char* key, SDL_Color& out,
                        const char* owner) {
  const YAML::Node field = node[key];
  if (!field) {
    return;
  }

  if (!field.IsMap()) {
    std::cerr << "Invalid UI config value for " << owner << "." << key << ": "
              << "expected map\n";
    return;
  }

  DecodeColorChannel(field, "r", out.r, owner);
  DecodeColorChannel(field, "g", out.g, owner);
  DecodeColorChannel(field, "b", out.b, owner);
  DecodeColorChannel(field, "a", out.a, owner);
}

template <typename StructType, typename MemberType>
void DecodeMember(const YAML::Node& node, const char* key, StructType& out,
                  MemberType StructType::* member, const char* owner) {
  DecodeField(node, key, out.*member, owner);
}

}  // namespace arelto::config::detail

namespace YAML {

using ::arelto::config::detail::DecodeColorChannel;
using ::arelto::config::detail::DecodeMember;

template <>
struct convert<SDL_Color> {
  static Node encode(const SDL_Color& rhs) {
    Node node;
    node["r"] = static_cast<int>(rhs.r);
    node["g"] = static_cast<int>(rhs.g);
    node["b"] = static_cast<int>(rhs.b);
    node["a"] = static_cast<int>(rhs.a);
    return node;
  }

  static bool decode(const Node& node, SDL_Color& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "color";

    DecodeColorChannel(node, "r", rhs.r, kOwner);
    DecodeColorChannel(node, "g", rhs.g, kOwner);
    DecodeColorChannel(node, "b", rhs.b, kOwner);
    DecodeColorChannel(node, "a", rhs.a, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::UIColorsConfig> {
  static Node encode(const arelto::UIColorsConfig& rhs) {
    Node node;
    node["positive_green"] = rhs.positive_green;
    node["negative_red"] = rhs.negative_red;
    return node;
  }

  static bool decode(const Node& node, arelto::UIColorsConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "ui.colors";

    DecodeMember(node, "positive_green", rhs,
                 &arelto::UIColorsConfig::positive_green, kOwner);
    DecodeMember(node, "negative_red", rhs,
                 &arelto::UIColorsConfig::negative_red, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::UIFontConfig> {
  static Node encode(const arelto::UIFontConfig& rhs) {
    Node node;
    node["kFontSizeSmall"] = rhs.kFontSizeSmall;
    node["kFontSizeMedium"] = rhs.kFontSizeMedium;
    node["kFontSizeLarge"] = rhs.kFontSizeLarge;
    node["kFontSizeHuge"] = rhs.kFontSizeHuge;
    return node;
  }

  static bool decode(const Node& node, arelto::UIFontConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "ui.fonts";

    DecodeMember(node, "kFontSizeSmall", rhs,
                 &arelto::UIFontConfig::kFontSizeSmall, kOwner);
    DecodeMember(node, "kFontSizeMedium", rhs,
                 &arelto::UIFontConfig::kFontSizeMedium, kOwner);
    DecodeMember(node, "kFontSizeLarge", rhs,
                 &arelto::UIFontConfig::kFontSizeLarge, kOwner);
    DecodeMember(node, "kFontSizeHuge", rhs,
                 &arelto::UIFontConfig::kFontSizeHuge, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::UIHudConfig> {
  static Node encode(const arelto::UIHudConfig& rhs) {
    Node node;
    node["kHudPadding"] = rhs.kHudPadding;
    node["kHudBarSpacing"] = rhs.kHudBarSpacing;
    node["kBarTextOffsetX"] = rhs.kBarTextOffsetX;
    node["kBarTextOffsetY"] = rhs.kBarTextOffsetY;
    node["kLevelGroupOffsetY"] = rhs.kLevelGroupOffsetY;
    node["kTimerTextGap"] = rhs.kTimerTextGap;
    node["kLevelTextGap"] = rhs.kLevelTextGap;
    node["kDigitSpriteWidth"] = rhs.kDigitSpriteWidth;
    node["kDigitSpriteHeight"] = rhs.kDigitSpriteHeight;
    node["kHealthBarContainerSpriteOffsetX"] =
        rhs.kHealthBarContainerSpriteOffsetX;
    node["kHealthBarContainerSpriteOffsetY"] =
        rhs.kHealthBarContainerSpriteOffsetY;
    node["kHealthBarContainerSpriteWidth"] = rhs.kHealthBarContainerSpriteWidth;
    node["kHealthBarContainerSpriteHeight"] =
        rhs.kHealthBarContainerSpriteHeight;
    node["kHealthBarRelOffsetX"] = rhs.kHealthBarRelOffsetX;
    node["kHealthBarRelOffsetY"] = rhs.kHealthBarRelOffsetY;
    node["kHealthBarSpriteOffsetX"] = rhs.kHealthBarSpriteOffsetX;
    node["kHealthBarSpriteOffsetY"] = rhs.kHealthBarSpriteOffsetY;
    node["kHealthBarSpriteWidth"] = rhs.kHealthBarSpriteWidth;
    node["kHealthBarSpriteHeight"] = rhs.kHealthBarSpriteHeight;
    node["kHealthBarTextRelOffsetX"] = rhs.kHealthBarTextRelOffsetX;
    node["kHealthBarTextRelOffsetY"] = rhs.kHealthBarTextRelOffsetY;
    node["kHealthBarTextCharWidth"] = rhs.kHealthBarTextCharWidth;
    node["kHealthBarTextCharHeight"] = rhs.kHealthBarTextCharHeight;
    node["kTimerHourglassSpriteWidth"] = rhs.kTimerHourglassSpriteWidth;
    node["kTimerHourglassSpriteHeight"] = rhs.kTimerHourglassSpriteHeight;
    node["kTimerTextCharWidth"] = rhs.kTimerTextCharWidth;
    node["kTimerTextCharHeight"] = rhs.kTimerTextCharHeight;
    node["kGameOverSpriteWidth"] = rhs.kGameOverSpriteWidth;
    node["kGameOverSpriteHeight"] = rhs.kGameOverSpriteHeight;
    node["kExpBarContainerSpriteOffsetX"] = rhs.kExpBarContainerSpriteOffsetX;
    node["kExpBarContainerSpriteOffsetY"] = rhs.kExpBarContainerSpriteOffsetY;
    node["kExpBarContainerSpriteWidth"] = rhs.kExpBarContainerSpriteWidth;
    node["kExpBarContainerSpriteHeight"] = rhs.kExpBarContainerSpriteHeight;
    node["kExpBarRelOffsetX"] = rhs.kExpBarRelOffsetX;
    node["kExpBarRelOffsetY"] = rhs.kExpBarRelOffsetY;
    node["kExpBarSpriteOffsetX"] = rhs.kExpBarSpriteOffsetX;
    node["kExpBarSpriteOffsetY"] = rhs.kExpBarSpriteOffsetY;
    node["kExpBarSpriteWidth"] = rhs.kExpBarSpriteWidth;
    node["kExpBarSpriteHeight"] = rhs.kExpBarSpriteHeight;
    node["kExpBarTextRelOffsetX"] = rhs.kExpBarTextRelOffsetX;
    node["kExpBarTextRelOffsetY"] = rhs.kExpBarTextRelOffsetY;
    node["kExpBarTextCharWidth"] = rhs.kExpBarTextCharWidth;
    node["kExpBarTextCharHeight"] = rhs.kExpBarTextCharHeight;
    node["kLevelIconSpriteOffsetX"] = rhs.kLevelIconSpriteOffsetX;
    node["kLevelIconSpriteOffsetY"] = rhs.kLevelIconSpriteOffsetY;
    node["kLevelIconSpriteWidth"] = rhs.kLevelIconSpriteWidth;
    node["kLevelIconSpriteHeight"] = rhs.kLevelIconSpriteHeight;
    node["kLevelTextCharWidth"] = rhs.kLevelTextCharWidth;
    node["kLevelTextCharHeight"] = rhs.kLevelTextCharHeight;
    node["kLevelUpIconMargin"] = rhs.kLevelUpIconMargin;
    node["kLevelUpTextMargin"] = rhs.kLevelUpTextMargin;
    return node;
  }

  static bool decode(const Node& node, arelto::UIHudConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "ui.hud";

    DecodeMember(node, "kHudPadding", rhs, &arelto::UIHudConfig::kHudPadding,
                 kOwner);
    DecodeMember(node, "kHudBarSpacing", rhs,
                 &arelto::UIHudConfig::kHudBarSpacing, kOwner);
    DecodeMember(node, "kBarTextOffsetX", rhs,
                 &arelto::UIHudConfig::kBarTextOffsetX, kOwner);
    DecodeMember(node, "kBarTextOffsetY", rhs,
                 &arelto::UIHudConfig::kBarTextOffsetY, kOwner);
    DecodeMember(node, "kLevelGroupOffsetY", rhs,
                 &arelto::UIHudConfig::kLevelGroupOffsetY, kOwner);
    DecodeMember(node, "kTimerTextGap", rhs,
                 &arelto::UIHudConfig::kTimerTextGap, kOwner);
    DecodeMember(node, "kLevelTextGap", rhs,
                 &arelto::UIHudConfig::kLevelTextGap, kOwner);
    DecodeMember(node, "kDigitSpriteWidth", rhs,
                 &arelto::UIHudConfig::kDigitSpriteWidth, kOwner);
    DecodeMember(node, "kDigitSpriteHeight", rhs,
                 &arelto::UIHudConfig::kDigitSpriteHeight, kOwner);
    DecodeMember(node, "kHealthBarContainerSpriteOffsetX", rhs,
                 &arelto::UIHudConfig::kHealthBarContainerSpriteOffsetX,
                 kOwner);
    DecodeMember(node, "kHealthBarContainerSpriteOffsetY", rhs,
                 &arelto::UIHudConfig::kHealthBarContainerSpriteOffsetY,
                 kOwner);
    DecodeMember(node, "kHealthBarContainerSpriteWidth", rhs,
                 &arelto::UIHudConfig::kHealthBarContainerSpriteWidth, kOwner);
    DecodeMember(node, "kHealthBarContainerSpriteHeight", rhs,
                 &arelto::UIHudConfig::kHealthBarContainerSpriteHeight, kOwner);
    DecodeMember(node, "kHealthBarRelOffsetX", rhs,
                 &arelto::UIHudConfig::kHealthBarRelOffsetX, kOwner);
    DecodeMember(node, "kHealthBarRelOffsetY", rhs,
                 &arelto::UIHudConfig::kHealthBarRelOffsetY, kOwner);
    DecodeMember(node, "kHealthBarSpriteOffsetX", rhs,
                 &arelto::UIHudConfig::kHealthBarSpriteOffsetX, kOwner);
    DecodeMember(node, "kHealthBarSpriteOffsetY", rhs,
                 &arelto::UIHudConfig::kHealthBarSpriteOffsetY, kOwner);
    DecodeMember(node, "kHealthBarSpriteWidth", rhs,
                 &arelto::UIHudConfig::kHealthBarSpriteWidth, kOwner);
    DecodeMember(node, "kHealthBarSpriteHeight", rhs,
                 &arelto::UIHudConfig::kHealthBarSpriteHeight, kOwner);
    DecodeMember(node, "kHealthBarTextRelOffsetX", rhs,
                 &arelto::UIHudConfig::kHealthBarTextRelOffsetX, kOwner);
    DecodeMember(node, "kHealthBarTextRelOffsetY", rhs,
                 &arelto::UIHudConfig::kHealthBarTextRelOffsetY, kOwner);
    DecodeMember(node, "kHealthBarTextCharWidth", rhs,
                 &arelto::UIHudConfig::kHealthBarTextCharWidth, kOwner);
    DecodeMember(node, "kHealthBarTextCharHeight", rhs,
                 &arelto::UIHudConfig::kHealthBarTextCharHeight, kOwner);
    DecodeMember(node, "kTimerHourglassSpriteWidth", rhs,
                 &arelto::UIHudConfig::kTimerHourglassSpriteWidth, kOwner);
    DecodeMember(node, "kTimerHourglassSpriteHeight", rhs,
                 &arelto::UIHudConfig::kTimerHourglassSpriteHeight, kOwner);
    DecodeMember(node, "kTimerTextCharWidth", rhs,
                 &arelto::UIHudConfig::kTimerTextCharWidth, kOwner);
    DecodeMember(node, "kTimerTextCharHeight", rhs,
                 &arelto::UIHudConfig::kTimerTextCharHeight, kOwner);
    DecodeMember(node, "kGameOverSpriteWidth", rhs,
                 &arelto::UIHudConfig::kGameOverSpriteWidth, kOwner);
    DecodeMember(node, "kGameOverSpriteHeight", rhs,
                 &arelto::UIHudConfig::kGameOverSpriteHeight, kOwner);
    DecodeMember(node, "kExpBarContainerSpriteOffsetX", rhs,
                 &arelto::UIHudConfig::kExpBarContainerSpriteOffsetX, kOwner);
    DecodeMember(node, "kExpBarContainerSpriteOffsetY", rhs,
                 &arelto::UIHudConfig::kExpBarContainerSpriteOffsetY, kOwner);
    DecodeMember(node, "kExpBarContainerSpriteWidth", rhs,
                 &arelto::UIHudConfig::kExpBarContainerSpriteWidth, kOwner);
    DecodeMember(node, "kExpBarContainerSpriteHeight", rhs,
                 &arelto::UIHudConfig::kExpBarContainerSpriteHeight, kOwner);
    DecodeMember(node, "kExpBarRelOffsetX", rhs,
                 &arelto::UIHudConfig::kExpBarRelOffsetX, kOwner);
    DecodeMember(node, "kExpBarRelOffsetY", rhs,
                 &arelto::UIHudConfig::kExpBarRelOffsetY, kOwner);
    DecodeMember(node, "kExpBarSpriteOffsetX", rhs,
                 &arelto::UIHudConfig::kExpBarSpriteOffsetX, kOwner);
    DecodeMember(node, "kExpBarSpriteOffsetY", rhs,
                 &arelto::UIHudConfig::kExpBarSpriteOffsetY, kOwner);
    DecodeMember(node, "kExpBarSpriteWidth", rhs,
                 &arelto::UIHudConfig::kExpBarSpriteWidth, kOwner);
    DecodeMember(node, "kExpBarSpriteHeight", rhs,
                 &arelto::UIHudConfig::kExpBarSpriteHeight, kOwner);
    DecodeMember(node, "kExpBarTextRelOffsetX", rhs,
                 &arelto::UIHudConfig::kExpBarTextRelOffsetX, kOwner);
    DecodeMember(node, "kExpBarTextRelOffsetY", rhs,
                 &arelto::UIHudConfig::kExpBarTextRelOffsetY, kOwner);
    DecodeMember(node, "kExpBarTextCharWidth", rhs,
                 &arelto::UIHudConfig::kExpBarTextCharWidth, kOwner);
    DecodeMember(node, "kExpBarTextCharHeight", rhs,
                 &arelto::UIHudConfig::kExpBarTextCharHeight, kOwner);
    DecodeMember(node, "kLevelIconSpriteOffsetX", rhs,
                 &arelto::UIHudConfig::kLevelIconSpriteOffsetX, kOwner);
    DecodeMember(node, "kLevelIconSpriteOffsetY", rhs,
                 &arelto::UIHudConfig::kLevelIconSpriteOffsetY, kOwner);
    DecodeMember(node, "kLevelIconSpriteWidth", rhs,
                 &arelto::UIHudConfig::kLevelIconSpriteWidth, kOwner);
    DecodeMember(node, "kLevelIconSpriteHeight", rhs,
                 &arelto::UIHudConfig::kLevelIconSpriteHeight, kOwner);
    DecodeMember(node, "kLevelTextCharWidth", rhs,
                 &arelto::UIHudConfig::kLevelTextCharWidth, kOwner);
    DecodeMember(node, "kLevelTextCharHeight", rhs,
                 &arelto::UIHudConfig::kLevelTextCharHeight, kOwner);
    DecodeMember(node, "kLevelUpIconMargin", rhs,
                 &arelto::UIHudConfig::kLevelUpIconMargin, kOwner);
    DecodeMember(node, "kLevelUpTextMargin", rhs,
                 &arelto::UIHudConfig::kLevelUpTextMargin, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::UIMenuConfig> {
  static Node encode(const arelto::UIMenuConfig& rhs) {
    Node node;
    node["kMenuContentPadding"] = rhs.kMenuContentPadding;
    node["kMenuItemSpacing"] = rhs.kMenuItemSpacing;
    node["kMenuButtonGap"] = rhs.kMenuButtonGap;
    node["kMenuBottomPadding"] = rhs.kMenuBottomPadding;
    node["kGenericButtonTextureWidth"] = rhs.kGenericButtonTextureWidth;
    node["kGenericButtonTextureHeight"] = rhs.kGenericButtonTextureHeight;
    node["kBeginButtonTextureWidth"] = rhs.kBeginButtonTextureWidth;
    node["kBeginButtonTextureHeight"] = rhs.kBeginButtonTextureHeight;
    node["kBeginButtonWidth"] = rhs.kBeginButtonWidth;
    node["kBeginButtonHeight"] = rhs.kBeginButtonHeight;
    node["kBeginButtonY"] = rhs.kBeginButtonY;
    node["kSettingsMenuWidth"] = rhs.kSettingsMenuWidth;
    node["kSettingsMenuHeight"] = rhs.kSettingsMenuHeight;
    node["kSettingsMenuBackgroundSpriteWidth"] =
        rhs.kSettingsMenuBackgroundSpriteWidth;
    node["kSettingsMenuBackgroundSpriteHeight"] =
        rhs.kSettingsMenuBackgroundSpriteHeight;
    node["kSettingsMenuButtonWidth"] = rhs.kSettingsMenuButtonWidth;
    node["kSettingsMenuButtonHeight"] = rhs.kSettingsMenuButtonHeight;
    node["kSettingsMenuVolumeSliderWidth"] = rhs.kSettingsMenuVolumeSliderWidth;
    node["kSettingsMenuVolumeSliderHeight"] =
        rhs.kSettingsMenuVolumeSliderHeight;
    node["kVolumeSliderFillOffsetX"] = rhs.kVolumeSliderFillOffsetX;
    node["kVolumeSliderFillOffsetY"] = rhs.kVolumeSliderFillOffsetY;
    node["kVolumeSliderFillWidth"] = rhs.kVolumeSliderFillWidth;
    node["kVolumeSliderFillHeight"] = rhs.kVolumeSliderFillHeight;
    node["kQuitMenuWidth"] = rhs.kQuitMenuWidth;
    node["kQuitMenuHeight"] = rhs.kQuitMenuHeight;
    node["kSliderContainerSpriteOffsetX"] = rhs.kSliderContainerSpriteOffsetX;
    node["kSliderContainerSpriteOffsetY"] = rhs.kSliderContainerSpriteOffsetY;
    node["kSliderContainerSpriteWidth"] = rhs.kSliderContainerSpriteWidth;
    node["kSliderContainerSpriteHeight"] = rhs.kSliderContainerSpriteHeight;
    node["kSliderBarSpriteOffsetX"] = rhs.kSliderBarSpriteOffsetX;
    node["kSliderBarSpriteOffsetY"] = rhs.kSliderBarSpriteOffsetY;
    node["kSliderBarSpriteWidth"] = rhs.kSliderBarSpriteWidth;
    node["kSliderBarSpriteHeight"] = rhs.kSliderBarSpriteHeight;
    node["kCheckboxSpriteWidth"] = rhs.kCheckboxSpriteWidth;
    node["kCheckboxSpriteHeight"] = rhs.kCheckboxSpriteHeight;
    node["kCheckmarkSpriteWidth"] = rhs.kCheckmarkSpriteWidth;
    node["kCheckmarkSpriteHeight"] = rhs.kCheckmarkSpriteHeight;
    return node;
  }

  static bool decode(const Node& node, arelto::UIMenuConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "ui.menus";

    DecodeMember(node, "kMenuContentPadding", rhs,
                 &arelto::UIMenuConfig::kMenuContentPadding, kOwner);
    DecodeMember(node, "kMenuItemSpacing", rhs,
                 &arelto::UIMenuConfig::kMenuItemSpacing, kOwner);
    DecodeMember(node, "kMenuButtonGap", rhs,
                 &arelto::UIMenuConfig::kMenuButtonGap, kOwner);
    DecodeMember(node, "kMenuBottomPadding", rhs,
                 &arelto::UIMenuConfig::kMenuBottomPadding, kOwner);
    DecodeMember(node, "kGenericButtonTextureWidth", rhs,
                 &arelto::UIMenuConfig::kGenericButtonTextureWidth, kOwner);
    DecodeMember(node, "kGenericButtonTextureHeight", rhs,
                 &arelto::UIMenuConfig::kGenericButtonTextureHeight, kOwner);
    DecodeMember(node, "kBeginButtonTextureWidth", rhs,
                 &arelto::UIMenuConfig::kBeginButtonTextureWidth, kOwner);
    DecodeMember(node, "kBeginButtonTextureHeight", rhs,
                 &arelto::UIMenuConfig::kBeginButtonTextureHeight, kOwner);
    DecodeMember(node, "kBeginButtonWidth", rhs,
                 &arelto::UIMenuConfig::kBeginButtonWidth, kOwner);
    DecodeMember(node, "kBeginButtonHeight", rhs,
                 &arelto::UIMenuConfig::kBeginButtonHeight, kOwner);
    DecodeMember(node, "kBeginButtonY", rhs,
                 &arelto::UIMenuConfig::kBeginButtonY, kOwner);
    DecodeMember(node, "kSettingsMenuWidth", rhs,
                 &arelto::UIMenuConfig::kSettingsMenuWidth, kOwner);
    DecodeMember(node, "kSettingsMenuHeight", rhs,
                 &arelto::UIMenuConfig::kSettingsMenuHeight, kOwner);
    DecodeMember(node, "kSettingsMenuBackgroundSpriteWidth", rhs,
                 &arelto::UIMenuConfig::kSettingsMenuBackgroundSpriteWidth,
                 kOwner);
    DecodeMember(node, "kSettingsMenuBackgroundSpriteHeight", rhs,
                 &arelto::UIMenuConfig::kSettingsMenuBackgroundSpriteHeight,
                 kOwner);
    DecodeMember(node, "kSettingsMenuButtonWidth", rhs,
                 &arelto::UIMenuConfig::kSettingsMenuButtonWidth, kOwner);
    DecodeMember(node, "kSettingsMenuButtonHeight", rhs,
                 &arelto::UIMenuConfig::kSettingsMenuButtonHeight, kOwner);
    DecodeMember(node, "kSettingsMenuVolumeSliderWidth", rhs,
                 &arelto::UIMenuConfig::kSettingsMenuVolumeSliderWidth, kOwner);
    DecodeMember(node, "kSettingsMenuVolumeSliderHeight", rhs,
                 &arelto::UIMenuConfig::kSettingsMenuVolumeSliderHeight,
                 kOwner);
    DecodeMember(node, "kVolumeSliderFillOffsetX", rhs,
                 &arelto::UIMenuConfig::kVolumeSliderFillOffsetX, kOwner);
    DecodeMember(node, "kVolumeSliderFillOffsetY", rhs,
                 &arelto::UIMenuConfig::kVolumeSliderFillOffsetY, kOwner);
    DecodeMember(node, "kVolumeSliderFillWidth", rhs,
                 &arelto::UIMenuConfig::kVolumeSliderFillWidth, kOwner);
    DecodeMember(node, "kVolumeSliderFillHeight", rhs,
                 &arelto::UIMenuConfig::kVolumeSliderFillHeight, kOwner);
    DecodeMember(node, "kQuitMenuWidth", rhs,
                 &arelto::UIMenuConfig::kQuitMenuWidth, kOwner);
    DecodeMember(node, "kQuitMenuHeight", rhs,
                 &arelto::UIMenuConfig::kQuitMenuHeight, kOwner);
    DecodeMember(node, "kSliderContainerSpriteOffsetX", rhs,
                 &arelto::UIMenuConfig::kSliderContainerSpriteOffsetX, kOwner);
    DecodeMember(node, "kSliderContainerSpriteOffsetY", rhs,
                 &arelto::UIMenuConfig::kSliderContainerSpriteOffsetY, kOwner);
    DecodeMember(node, "kSliderContainerSpriteWidth", rhs,
                 &arelto::UIMenuConfig::kSliderContainerSpriteWidth, kOwner);
    DecodeMember(node, "kSliderContainerSpriteHeight", rhs,
                 &arelto::UIMenuConfig::kSliderContainerSpriteHeight, kOwner);
    DecodeMember(node, "kSliderBarSpriteOffsetX", rhs,
                 &arelto::UIMenuConfig::kSliderBarSpriteOffsetX, kOwner);
    DecodeMember(node, "kSliderBarSpriteOffsetY", rhs,
                 &arelto::UIMenuConfig::kSliderBarSpriteOffsetY, kOwner);
    DecodeMember(node, "kSliderBarSpriteWidth", rhs,
                 &arelto::UIMenuConfig::kSliderBarSpriteWidth, kOwner);
    DecodeMember(node, "kSliderBarSpriteHeight", rhs,
                 &arelto::UIMenuConfig::kSliderBarSpriteHeight, kOwner);
    DecodeMember(node, "kCheckboxSpriteWidth", rhs,
                 &arelto::UIMenuConfig::kCheckboxSpriteWidth, kOwner);
    DecodeMember(node, "kCheckboxSpriteHeight", rhs,
                 &arelto::UIMenuConfig::kCheckboxSpriteHeight, kOwner);
    DecodeMember(node, "kCheckmarkSpriteWidth", rhs,
                 &arelto::UIMenuConfig::kCheckmarkSpriteWidth, kOwner);
    DecodeMember(node, "kCheckmarkSpriteHeight", rhs,
                 &arelto::UIMenuConfig::kCheckmarkSpriteHeight, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::UICardConfig> {
  static Node encode(const arelto::UICardConfig& rhs) {
    Node node;
    node["kLevelUpCardWidth"] = rhs.kLevelUpCardWidth;
    node["kLevelUpCardHeight"] = rhs.kLevelUpCardHeight;
    node["kLevelUpCardGap"] = rhs.kLevelUpCardGap;
    node["kLevelUpIconOffsetY"] = rhs.kLevelUpIconOffsetY;
    node["kLevelUpIconSize"] = rhs.kLevelUpIconSize;
    node["kLevelUpNameOffsetY"] = rhs.kLevelUpNameOffsetY;
    node["kLevelUpNameOffsetX"] = rhs.kLevelUpNameOffsetX;
    node["kLevelUpDescOffsetY"] = rhs.kLevelUpDescOffsetY;
    node["kLevelUpDescOffsetX"] = rhs.kLevelUpDescOffsetX;
    node["kLevelUpStatsOffsetY"] = rhs.kLevelUpStatsOffsetY;
    node["kLevelUpStatsOffsetX"] = rhs.kLevelUpStatsOffsetX;
    node["kLevelUpRowStride"] = rhs.kLevelUpRowStride;
    node["kLevelUpButtonOffsetY"] = rhs.kLevelUpButtonOffsetY;
    node["kLevelUpButtonWidth"] = rhs.kLevelUpButtonWidth;
    node["kLevelUpButtonHeight"] = rhs.kLevelUpButtonHeight;
    node["kItemIconSize"] = rhs.kItemIconSize;
    node["kItemCardWidth"] = rhs.kItemCardWidth;
    node["kItemCardHeight"] = rhs.kItemCardHeight;
    node["kItemCardGap"] = rhs.kItemCardGap;
    node["kItemCardIconOffsetY"] = rhs.kItemCardIconOffsetY;
    node["kItemCardIconSize"] = rhs.kItemCardIconSize;
    node["kItemCardNameOffsetY"] = rhs.kItemCardNameOffsetY;
    node["kItemCardNameOffsetX"] = rhs.kItemCardNameOffsetX;
    node["kItemCardDescOffsetY"] = rhs.kItemCardDescOffsetY;
    node["kItemCardDescOffsetX"] = rhs.kItemCardDescOffsetX;
    node["kItemCardStatsOffsetY"] = rhs.kItemCardStatsOffsetY;
    node["kItemCardStatsOffsetX"] = rhs.kItemCardStatsOffsetX;
    node["kItemCardRowStride"] = rhs.kItemCardRowStride;
    node["kItemCardButtonOffsetY"] = rhs.kItemCardButtonOffsetY;
    node["kItemCardButtonWidth"] = rhs.kItemCardButtonWidth;
    node["kItemCardButtonHeight"] = rhs.kItemCardButtonHeight;
    return node;
  }

  static bool decode(const Node& node, arelto::UICardConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "ui.cards";

    DecodeMember(node, "kLevelUpCardWidth", rhs,
                 &arelto::UICardConfig::kLevelUpCardWidth, kOwner);
    DecodeMember(node, "kLevelUpCardHeight", rhs,
                 &arelto::UICardConfig::kLevelUpCardHeight, kOwner);
    DecodeMember(node, "kLevelUpCardGap", rhs,
                 &arelto::UICardConfig::kLevelUpCardGap, kOwner);
    DecodeMember(node, "kLevelUpIconOffsetY", rhs,
                 &arelto::UICardConfig::kLevelUpIconOffsetY, kOwner);
    DecodeMember(node, "kLevelUpIconSize", rhs,
                 &arelto::UICardConfig::kLevelUpIconSize, kOwner);
    DecodeMember(node, "kLevelUpNameOffsetY", rhs,
                 &arelto::UICardConfig::kLevelUpNameOffsetY, kOwner);
    DecodeMember(node, "kLevelUpNameOffsetX", rhs,
                 &arelto::UICardConfig::kLevelUpNameOffsetX, kOwner);
    DecodeMember(node, "kLevelUpDescOffsetY", rhs,
                 &arelto::UICardConfig::kLevelUpDescOffsetY, kOwner);
    DecodeMember(node, "kLevelUpDescOffsetX", rhs,
                 &arelto::UICardConfig::kLevelUpDescOffsetX, kOwner);
    DecodeMember(node, "kLevelUpStatsOffsetY", rhs,
                 &arelto::UICardConfig::kLevelUpStatsOffsetY, kOwner);
    DecodeMember(node, "kLevelUpStatsOffsetX", rhs,
                 &arelto::UICardConfig::kLevelUpStatsOffsetX, kOwner);
    DecodeMember(node, "kLevelUpRowStride", rhs,
                 &arelto::UICardConfig::kLevelUpRowStride, kOwner);
    DecodeMember(node, "kLevelUpButtonOffsetY", rhs,
                 &arelto::UICardConfig::kLevelUpButtonOffsetY, kOwner);
    DecodeMember(node, "kLevelUpButtonWidth", rhs,
                 &arelto::UICardConfig::kLevelUpButtonWidth, kOwner);
    DecodeMember(node, "kLevelUpButtonHeight", rhs,
                 &arelto::UICardConfig::kLevelUpButtonHeight, kOwner);
    DecodeMember(node, "kItemIconSize", rhs,
                 &arelto::UICardConfig::kItemIconSize, kOwner);
    DecodeMember(node, "kItemCardWidth", rhs,
                 &arelto::UICardConfig::kItemCardWidth, kOwner);
    DecodeMember(node, "kItemCardHeight", rhs,
                 &arelto::UICardConfig::kItemCardHeight, kOwner);
    DecodeMember(node, "kItemCardGap", rhs, &arelto::UICardConfig::kItemCardGap,
                 kOwner);
    DecodeMember(node, "kItemCardIconOffsetY", rhs,
                 &arelto::UICardConfig::kItemCardIconOffsetY, kOwner);
    DecodeMember(node, "kItemCardIconSize", rhs,
                 &arelto::UICardConfig::kItemCardIconSize, kOwner);
    DecodeMember(node, "kItemCardNameOffsetY", rhs,
                 &arelto::UICardConfig::kItemCardNameOffsetY, kOwner);
    DecodeMember(node, "kItemCardNameOffsetX", rhs,
                 &arelto::UICardConfig::kItemCardNameOffsetX, kOwner);
    DecodeMember(node, "kItemCardDescOffsetY", rhs,
                 &arelto::UICardConfig::kItemCardDescOffsetY, kOwner);
    DecodeMember(node, "kItemCardDescOffsetX", rhs,
                 &arelto::UICardConfig::kItemCardDescOffsetX, kOwner);
    DecodeMember(node, "kItemCardStatsOffsetY", rhs,
                 &arelto::UICardConfig::kItemCardStatsOffsetY, kOwner);
    DecodeMember(node, "kItemCardStatsOffsetX", rhs,
                 &arelto::UICardConfig::kItemCardStatsOffsetX, kOwner);
    DecodeMember(node, "kItemCardRowStride", rhs,
                 &arelto::UICardConfig::kItemCardRowStride, kOwner);
    DecodeMember(node, "kItemCardButtonOffsetY", rhs,
                 &arelto::UICardConfig::kItemCardButtonOffsetY, kOwner);
    DecodeMember(node, "kItemCardButtonWidth", rhs,
                 &arelto::UICardConfig::kItemCardButtonWidth, kOwner);
    DecodeMember(node, "kItemCardButtonHeight", rhs,
                 &arelto::UICardConfig::kItemCardButtonHeight, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::UIInventoryConfig> {
  static Node encode(const arelto::UIInventoryConfig& rhs) {
    Node node;
    node["kInventoryBarY"] = rhs.kInventoryBarY;
    node["kInventoryIconSize"] = rhs.kInventoryIconSize;
    node["kInventoryWidgetHeight"] = rhs.kInventoryWidgetHeight;
    node["kInventoryLabelWidth"] = rhs.kInventoryLabelWidth;
    node["kInventoryItemGap"] = rhs.kInventoryItemGap;
    node["kInventoryMultiplierSize"] = rhs.kInventoryMultiplierSize;
    node["kInventoryMultiplierMargin"] = rhs.kInventoryMultiplierMargin;
    node["kInventoryContainerPadding"] = rhs.kInventoryContainerPadding;
    node["kInventoryBackgroundAlpha"] = rhs.kInventoryBackgroundAlpha;
    return node;
  }

  static bool decode(const Node& node, arelto::UIInventoryConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "ui.inventory";

    DecodeMember(node, "kInventoryBarY", rhs,
                 &arelto::UIInventoryConfig::kInventoryBarY, kOwner);
    DecodeMember(node, "kInventoryIconSize", rhs,
                 &arelto::UIInventoryConfig::kInventoryIconSize, kOwner);
    DecodeMember(node, "kInventoryWidgetHeight", rhs,
                 &arelto::UIInventoryConfig::kInventoryWidgetHeight, kOwner);
    DecodeMember(node, "kInventoryLabelWidth", rhs,
                 &arelto::UIInventoryConfig::kInventoryLabelWidth, kOwner);
    DecodeMember(node, "kInventoryItemGap", rhs,
                 &arelto::UIInventoryConfig::kInventoryItemGap, kOwner);
    DecodeMember(node, "kInventoryMultiplierSize", rhs,
                 &arelto::UIInventoryConfig::kInventoryMultiplierSize, kOwner);
    DecodeMember(node, "kInventoryMultiplierMargin", rhs,
                 &arelto::UIInventoryConfig::kInventoryMultiplierMargin,
                 kOwner);
    DecodeMember(node, "kInventoryContainerPadding", rhs,
                 &arelto::UIInventoryConfig::kInventoryContainerPadding,
                 kOwner);
    DecodeMember(node, "kInventoryBackgroundAlpha", rhs,
                 &arelto::UIInventoryConfig::kInventoryBackgroundAlpha, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::UIConfig> {
  static Node encode(const arelto::UIConfig& rhs) {
    Node node;
    node["colors"] = rhs.colors;
    node["fonts"] = rhs.fonts;
    node["hud"] = rhs.hud;
    node["menus"] = rhs.menus;
    node["cards"] = rhs.cards;
    node["inventory"] = rhs.inventory;
    return node;
  }

  static bool decode(const Node& node, arelto::UIConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "ui";

    DecodeMember(node, "colors", rhs, &arelto::UIConfig::colors, kOwner);
    DecodeMember(node, "fonts", rhs, &arelto::UIConfig::fonts, kOwner);
    DecodeMember(node, "hud", rhs, &arelto::UIConfig::hud, kOwner);
    DecodeMember(node, "menus", rhs, &arelto::UIConfig::menus, kOwner);
    DecodeMember(node, "cards", rhs, &arelto::UIConfig::cards, kOwner);
    DecodeMember(node, "inventory", rhs, &arelto::UIConfig::inventory, kOwner);
    return true;
  }
};

}  // namespace YAML

#endif
