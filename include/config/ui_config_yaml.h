#ifndef RL2_CONFIG_UI_CONFIG_YAML_H_
#define RL2_CONFIG_UI_CONFIG_YAML_H_

#include <iostream>
#include "config/config_decoding.h"
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

template <>
struct FieldDecoder<SDL_Color> {
  static void Decode(const YAML::Node& node, const char* key, SDL_Color& out,
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
};

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
    node["font_size_small"] = rhs.font_size_small;
    node["font_size_medium"] = rhs.font_size_medium;
    node["font_size_large"] = rhs.font_size_large;
    node["font_size_huge"] = rhs.font_size_huge;
    return node;
  }

  static bool decode(const Node& node, arelto::UIFontConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "ui.fonts";

    DecodeMember(node, "font_size_small", rhs,
                 &arelto::UIFontConfig::font_size_small, kOwner);
    DecodeMember(node, "font_size_medium", rhs,
                 &arelto::UIFontConfig::font_size_medium, kOwner);
    DecodeMember(node, "font_size_large", rhs,
                 &arelto::UIFontConfig::font_size_large, kOwner);
    DecodeMember(node, "font_size_huge", rhs,
                 &arelto::UIFontConfig::font_size_huge, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::UIHudConfig> {
  static Node encode(const arelto::UIHudConfig& rhs) {
    Node node;
    node["hud_padding"] = rhs.hud_padding;
    node["hud_bar_spacing"] = rhs.hud_bar_spacing;
    node["bar_text_offset_x"] = rhs.bar_text_offset_x;
    node["bar_text_offset_y"] = rhs.bar_text_offset_y;
    node["level_group_offset_y"] = rhs.level_group_offset_y;
    node["timer_text_gap"] = rhs.timer_text_gap;
    node["level_text_gap"] = rhs.level_text_gap;
    node["digit_sprite_width"] = rhs.digit_sprite_width;
    node["digit_sprite_height"] = rhs.digit_sprite_height;
    node["health_bar_container_sprite_offset_x"] =
        rhs.health_bar_container_sprite_offset_x;
    node["health_bar_container_sprite_offset_y"] =
        rhs.health_bar_container_sprite_offset_y;
    node["health_bar_container_sprite_width"] =
        rhs.health_bar_container_sprite_width;
    node["health_bar_container_sprite_height"] =
        rhs.health_bar_container_sprite_height;
    node["health_bar_rel_offset_x"] = rhs.health_bar_rel_offset_x;
    node["health_bar_rel_offset_y"] = rhs.health_bar_rel_offset_y;
    node["health_bar_sprite_offset_x"] = rhs.health_bar_sprite_offset_x;
    node["health_bar_sprite_offset_y"] = rhs.health_bar_sprite_offset_y;
    node["health_bar_sprite_width"] = rhs.health_bar_sprite_width;
    node["health_bar_sprite_height"] = rhs.health_bar_sprite_height;
    node["health_bar_text_rel_offset_x"] = rhs.health_bar_text_rel_offset_x;
    node["health_bar_text_rel_offset_y"] = rhs.health_bar_text_rel_offset_y;
    node["health_bar_text_char_width"] = rhs.health_bar_text_char_width;
    node["health_bar_text_char_height"] = rhs.health_bar_text_char_height;
    node["timer_hourglass_sprite_width"] = rhs.timer_hourglass_sprite_width;
    node["timer_hourglass_sprite_height"] = rhs.timer_hourglass_sprite_height;
    node["timer_text_char_width"] = rhs.timer_text_char_width;
    node["timer_text_char_height"] = rhs.timer_text_char_height;
    node["game_over_sprite_width"] = rhs.game_over_sprite_width;
    node["game_over_sprite_height"] = rhs.game_over_sprite_height;
    node["exp_bar_container_sprite_offset_x"] =
        rhs.exp_bar_container_sprite_offset_x;
    node["exp_bar_container_sprite_offset_y"] =
        rhs.exp_bar_container_sprite_offset_y;
    node["exp_bar_container_sprite_width"] = rhs.exp_bar_container_sprite_width;
    node["exp_bar_container_sprite_height"] =
        rhs.exp_bar_container_sprite_height;
    node["exp_bar_rel_offset_x"] = rhs.exp_bar_rel_offset_x;
    node["exp_bar_rel_offset_y"] = rhs.exp_bar_rel_offset_y;
    node["exp_bar_sprite_offset_x"] = rhs.exp_bar_sprite_offset_x;
    node["exp_bar_sprite_offset_y"] = rhs.exp_bar_sprite_offset_y;
    node["exp_bar_sprite_width"] = rhs.exp_bar_sprite_width;
    node["exp_bar_sprite_height"] = rhs.exp_bar_sprite_height;
    node["exp_bar_text_rel_offset_x"] = rhs.exp_bar_text_rel_offset_x;
    node["exp_bar_text_rel_offset_y"] = rhs.exp_bar_text_rel_offset_y;
    node["exp_bar_text_char_width"] = rhs.exp_bar_text_char_width;
    node["exp_bar_text_char_height"] = rhs.exp_bar_text_char_height;
    node["level_icon_sprite_offset_x"] = rhs.level_icon_sprite_offset_x;
    node["level_icon_sprite_offset_y"] = rhs.level_icon_sprite_offset_y;
    node["level_icon_sprite_width"] = rhs.level_icon_sprite_width;
    node["level_icon_sprite_height"] = rhs.level_icon_sprite_height;
    node["level_text_char_width"] = rhs.level_text_char_width;
    node["level_text_char_height"] = rhs.level_text_char_height;
    node["level_up_icon_margin"] = rhs.level_up_icon_margin;
    node["level_up_text_margin"] = rhs.level_up_text_margin;
    return node;
  }

  static bool decode(const Node& node, arelto::UIHudConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "ui.hud";

    DecodeMember(node, "hud_padding", rhs, &arelto::UIHudConfig::hud_padding,
                 kOwner);
    DecodeMember(node, "hud_bar_spacing", rhs,
                 &arelto::UIHudConfig::hud_bar_spacing, kOwner);
    DecodeMember(node, "bar_text_offset_x", rhs,
                 &arelto::UIHudConfig::bar_text_offset_x, kOwner);
    DecodeMember(node, "bar_text_offset_y", rhs,
                 &arelto::UIHudConfig::bar_text_offset_y, kOwner);
    DecodeMember(node, "level_group_offset_y", rhs,
                 &arelto::UIHudConfig::level_group_offset_y, kOwner);
    DecodeMember(node, "timer_text_gap", rhs,
                 &arelto::UIHudConfig::timer_text_gap, kOwner);
    DecodeMember(node, "level_text_gap", rhs,
                 &arelto::UIHudConfig::level_text_gap, kOwner);
    DecodeMember(node, "digit_sprite_width", rhs,
                 &arelto::UIHudConfig::digit_sprite_width, kOwner);
    DecodeMember(node, "digit_sprite_height", rhs,
                 &arelto::UIHudConfig::digit_sprite_height, kOwner);
    DecodeMember(node, "health_bar_container_sprite_offset_x", rhs,
                 &arelto::UIHudConfig::health_bar_container_sprite_offset_x,
                 kOwner);
    DecodeMember(node, "health_bar_container_sprite_offset_y", rhs,
                 &arelto::UIHudConfig::health_bar_container_sprite_offset_y,
                 kOwner);
    DecodeMember(node, "health_bar_container_sprite_width", rhs,
                 &arelto::UIHudConfig::health_bar_container_sprite_width,
                 kOwner);
    DecodeMember(node, "health_bar_container_sprite_height", rhs,
                 &arelto::UIHudConfig::health_bar_container_sprite_height,
                 kOwner);
    DecodeMember(node, "health_bar_rel_offset_x", rhs,
                 &arelto::UIHudConfig::health_bar_rel_offset_x, kOwner);
    DecodeMember(node, "health_bar_rel_offset_y", rhs,
                 &arelto::UIHudConfig::health_bar_rel_offset_y, kOwner);
    DecodeMember(node, "health_bar_sprite_offset_x", rhs,
                 &arelto::UIHudConfig::health_bar_sprite_offset_x, kOwner);
    DecodeMember(node, "health_bar_sprite_offset_y", rhs,
                 &arelto::UIHudConfig::health_bar_sprite_offset_y, kOwner);
    DecodeMember(node, "health_bar_sprite_width", rhs,
                 &arelto::UIHudConfig::health_bar_sprite_width, kOwner);
    DecodeMember(node, "health_bar_sprite_height", rhs,
                 &arelto::UIHudConfig::health_bar_sprite_height, kOwner);
    DecodeMember(node, "health_bar_text_rel_offset_x", rhs,
                 &arelto::UIHudConfig::health_bar_text_rel_offset_x, kOwner);
    DecodeMember(node, "health_bar_text_rel_offset_y", rhs,
                 &arelto::UIHudConfig::health_bar_text_rel_offset_y, kOwner);
    DecodeMember(node, "health_bar_text_char_width", rhs,
                 &arelto::UIHudConfig::health_bar_text_char_width, kOwner);
    DecodeMember(node, "health_bar_text_char_height", rhs,
                 &arelto::UIHudConfig::health_bar_text_char_height, kOwner);
    DecodeMember(node, "timer_hourglass_sprite_width", rhs,
                 &arelto::UIHudConfig::timer_hourglass_sprite_width, kOwner);
    DecodeMember(node, "timer_hourglass_sprite_height", rhs,
                 &arelto::UIHudConfig::timer_hourglass_sprite_height, kOwner);
    DecodeMember(node, "timer_text_char_width", rhs,
                 &arelto::UIHudConfig::timer_text_char_width, kOwner);
    DecodeMember(node, "timer_text_char_height", rhs,
                 &arelto::UIHudConfig::timer_text_char_height, kOwner);
    DecodeMember(node, "game_over_sprite_width", rhs,
                 &arelto::UIHudConfig::game_over_sprite_width, kOwner);
    DecodeMember(node, "game_over_sprite_height", rhs,
                 &arelto::UIHudConfig::game_over_sprite_height, kOwner);
    DecodeMember(node, "exp_bar_container_sprite_offset_x", rhs,
                 &arelto::UIHudConfig::exp_bar_container_sprite_offset_x,
                 kOwner);
    DecodeMember(node, "exp_bar_container_sprite_offset_y", rhs,
                 &arelto::UIHudConfig::exp_bar_container_sprite_offset_y,
                 kOwner);
    DecodeMember(node, "exp_bar_container_sprite_width", rhs,
                 &arelto::UIHudConfig::exp_bar_container_sprite_width, kOwner);
    DecodeMember(node, "exp_bar_container_sprite_height", rhs,
                 &arelto::UIHudConfig::exp_bar_container_sprite_height, kOwner);
    DecodeMember(node, "exp_bar_rel_offset_x", rhs,
                 &arelto::UIHudConfig::exp_bar_rel_offset_x, kOwner);
    DecodeMember(node, "exp_bar_rel_offset_y", rhs,
                 &arelto::UIHudConfig::exp_bar_rel_offset_y, kOwner);
    DecodeMember(node, "exp_bar_sprite_offset_x", rhs,
                 &arelto::UIHudConfig::exp_bar_sprite_offset_x, kOwner);
    DecodeMember(node, "exp_bar_sprite_offset_y", rhs,
                 &arelto::UIHudConfig::exp_bar_sprite_offset_y, kOwner);
    DecodeMember(node, "exp_bar_sprite_width", rhs,
                 &arelto::UIHudConfig::exp_bar_sprite_width, kOwner);
    DecodeMember(node, "exp_bar_sprite_height", rhs,
                 &arelto::UIHudConfig::exp_bar_sprite_height, kOwner);
    DecodeMember(node, "exp_bar_text_rel_offset_x", rhs,
                 &arelto::UIHudConfig::exp_bar_text_rel_offset_x, kOwner);
    DecodeMember(node, "exp_bar_text_rel_offset_y", rhs,
                 &arelto::UIHudConfig::exp_bar_text_rel_offset_y, kOwner);
    DecodeMember(node, "exp_bar_text_char_width", rhs,
                 &arelto::UIHudConfig::exp_bar_text_char_width, kOwner);
    DecodeMember(node, "exp_bar_text_char_height", rhs,
                 &arelto::UIHudConfig::exp_bar_text_char_height, kOwner);
    DecodeMember(node, "level_icon_sprite_offset_x", rhs,
                 &arelto::UIHudConfig::level_icon_sprite_offset_x, kOwner);
    DecodeMember(node, "level_icon_sprite_offset_y", rhs,
                 &arelto::UIHudConfig::level_icon_sprite_offset_y, kOwner);
    DecodeMember(node, "level_icon_sprite_width", rhs,
                 &arelto::UIHudConfig::level_icon_sprite_width, kOwner);
    DecodeMember(node, "level_icon_sprite_height", rhs,
                 &arelto::UIHudConfig::level_icon_sprite_height, kOwner);
    DecodeMember(node, "level_text_char_width", rhs,
                 &arelto::UIHudConfig::level_text_char_width, kOwner);
    DecodeMember(node, "level_text_char_height", rhs,
                 &arelto::UIHudConfig::level_text_char_height, kOwner);
    DecodeMember(node, "level_up_icon_margin", rhs,
                 &arelto::UIHudConfig::level_up_icon_margin, kOwner);
    DecodeMember(node, "level_up_text_margin", rhs,
                 &arelto::UIHudConfig::level_up_text_margin, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::UIMenuConfig> {
  static Node encode(const arelto::UIMenuConfig& rhs) {
    Node node;
    node["menu_content_padding"] = rhs.menu_content_padding;
    node["menu_item_spacing"] = rhs.menu_item_spacing;
    node["menu_button_gap"] = rhs.menu_button_gap;
    node["menu_bottom_padding"] = rhs.menu_bottom_padding;
    node["generic_button_texture_width"] = rhs.generic_button_texture_width;
    node["generic_button_texture_height"] = rhs.generic_button_texture_height;
    node["begin_button_texture_width"] = rhs.begin_button_texture_width;
    node["begin_button_texture_height"] = rhs.begin_button_texture_height;
    node["begin_button_width"] = rhs.begin_button_width;
    node["begin_button_height"] = rhs.begin_button_height;
    node["begin_button_y"] = rhs.begin_button_y;
    node["settings_menu_width"] = rhs.settings_menu_width;
    node["settings_menu_height"] = rhs.settings_menu_height;
    node["settings_menu_background_sprite_width"] =
        rhs.settings_menu_background_sprite_width;
    node["settings_menu_background_sprite_height"] =
        rhs.settings_menu_background_sprite_height;
    node["settings_menu_button_width"] = rhs.settings_menu_button_width;
    node["settings_menu_button_height"] = rhs.settings_menu_button_height;
    node["settings_menu_volume_slider_width"] =
        rhs.settings_menu_volume_slider_width;
    node["settings_menu_volume_slider_height"] =
        rhs.settings_menu_volume_slider_height;
    node["volume_slider_fill_offset_x"] = rhs.volume_slider_fill_offset_x;
    node["volume_slider_fill_offset_y"] = rhs.volume_slider_fill_offset_y;
    node["volume_slider_fill_width"] = rhs.volume_slider_fill_width;
    node["volume_slider_fill_height"] = rhs.volume_slider_fill_height;
    node["quit_menu_width"] = rhs.quit_menu_width;
    node["quit_menu_height"] = rhs.quit_menu_height;
    node["slider_container_sprite_offset_x"] =
        rhs.slider_container_sprite_offset_x;
    node["slider_container_sprite_offset_y"] =
        rhs.slider_container_sprite_offset_y;
    node["slider_container_sprite_width"] = rhs.slider_container_sprite_width;
    node["slider_container_sprite_height"] = rhs.slider_container_sprite_height;
    node["slider_bar_sprite_offset_x"] = rhs.slider_bar_sprite_offset_x;
    node["slider_bar_sprite_offset_y"] = rhs.slider_bar_sprite_offset_y;
    node["slider_bar_sprite_width"] = rhs.slider_bar_sprite_width;
    node["slider_bar_sprite_height"] = rhs.slider_bar_sprite_height;
    node["checkbox_sprite_width"] = rhs.checkbox_sprite_width;
    node["checkbox_sprite_height"] = rhs.checkbox_sprite_height;
    node["checkmark_sprite_width"] = rhs.checkmark_sprite_width;
    node["checkmark_sprite_height"] = rhs.checkmark_sprite_height;
    return node;
  }

  static bool decode(const Node& node, arelto::UIMenuConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "ui.menus";

    DecodeMember(node, "menu_content_padding", rhs,
                 &arelto::UIMenuConfig::menu_content_padding, kOwner);
    DecodeMember(node, "menu_item_spacing", rhs,
                 &arelto::UIMenuConfig::menu_item_spacing, kOwner);
    DecodeMember(node, "menu_button_gap", rhs,
                 &arelto::UIMenuConfig::menu_button_gap, kOwner);
    DecodeMember(node, "menu_bottom_padding", rhs,
                 &arelto::UIMenuConfig::menu_bottom_padding, kOwner);
    DecodeMember(node, "generic_button_texture_width", rhs,
                 &arelto::UIMenuConfig::generic_button_texture_width, kOwner);
    DecodeMember(node, "generic_button_texture_height", rhs,
                 &arelto::UIMenuConfig::generic_button_texture_height, kOwner);
    DecodeMember(node, "begin_button_texture_width", rhs,
                 &arelto::UIMenuConfig::begin_button_texture_width, kOwner);
    DecodeMember(node, "begin_button_texture_height", rhs,
                 &arelto::UIMenuConfig::begin_button_texture_height, kOwner);
    DecodeMember(node, "begin_button_width", rhs,
                 &arelto::UIMenuConfig::begin_button_width, kOwner);
    DecodeMember(node, "begin_button_height", rhs,
                 &arelto::UIMenuConfig::begin_button_height, kOwner);
    DecodeMember(node, "begin_button_y", rhs,
                 &arelto::UIMenuConfig::begin_button_y, kOwner);
    DecodeMember(node, "settings_menu_width", rhs,
                 &arelto::UIMenuConfig::settings_menu_width, kOwner);
    DecodeMember(node, "settings_menu_height", rhs,
                 &arelto::UIMenuConfig::settings_menu_height, kOwner);
    DecodeMember(node, "settings_menu_background_sprite_width", rhs,
                 &arelto::UIMenuConfig::settings_menu_background_sprite_width,
                 kOwner);
    DecodeMember(node, "settings_menu_background_sprite_height", rhs,
                 &arelto::UIMenuConfig::settings_menu_background_sprite_height,
                 kOwner);
    DecodeMember(node, "settings_menu_button_width", rhs,
                 &arelto::UIMenuConfig::settings_menu_button_width, kOwner);
    DecodeMember(node, "settings_menu_button_height", rhs,
                 &arelto::UIMenuConfig::settings_menu_button_height, kOwner);
    DecodeMember(node, "settings_menu_volume_slider_width", rhs,
                 &arelto::UIMenuConfig::settings_menu_volume_slider_width,
                 kOwner);
    DecodeMember(node, "settings_menu_volume_slider_height", rhs,
                 &arelto::UIMenuConfig::settings_menu_volume_slider_height,
                 kOwner);
    DecodeMember(node, "volume_slider_fill_offset_x", rhs,
                 &arelto::UIMenuConfig::volume_slider_fill_offset_x, kOwner);
    DecodeMember(node, "volume_slider_fill_offset_y", rhs,
                 &arelto::UIMenuConfig::volume_slider_fill_offset_y, kOwner);
    DecodeMember(node, "volume_slider_fill_width", rhs,
                 &arelto::UIMenuConfig::volume_slider_fill_width, kOwner);
    DecodeMember(node, "volume_slider_fill_height", rhs,
                 &arelto::UIMenuConfig::volume_slider_fill_height, kOwner);
    DecodeMember(node, "quit_menu_width", rhs,
                 &arelto::UIMenuConfig::quit_menu_width, kOwner);
    DecodeMember(node, "quit_menu_height", rhs,
                 &arelto::UIMenuConfig::quit_menu_height, kOwner);
    DecodeMember(node, "slider_container_sprite_offset_x", rhs,
                 &arelto::UIMenuConfig::slider_container_sprite_offset_x,
                 kOwner);
    DecodeMember(node, "slider_container_sprite_offset_y", rhs,
                 &arelto::UIMenuConfig::slider_container_sprite_offset_y,
                 kOwner);
    DecodeMember(node, "slider_container_sprite_width", rhs,
                 &arelto::UIMenuConfig::slider_container_sprite_width, kOwner);
    DecodeMember(node, "slider_container_sprite_height", rhs,
                 &arelto::UIMenuConfig::slider_container_sprite_height, kOwner);
    DecodeMember(node, "slider_bar_sprite_offset_x", rhs,
                 &arelto::UIMenuConfig::slider_bar_sprite_offset_x, kOwner);
    DecodeMember(node, "slider_bar_sprite_offset_y", rhs,
                 &arelto::UIMenuConfig::slider_bar_sprite_offset_y, kOwner);
    DecodeMember(node, "slider_bar_sprite_width", rhs,
                 &arelto::UIMenuConfig::slider_bar_sprite_width, kOwner);
    DecodeMember(node, "slider_bar_sprite_height", rhs,
                 &arelto::UIMenuConfig::slider_bar_sprite_height, kOwner);
    DecodeMember(node, "checkbox_sprite_width", rhs,
                 &arelto::UIMenuConfig::checkbox_sprite_width, kOwner);
    DecodeMember(node, "checkbox_sprite_height", rhs,
                 &arelto::UIMenuConfig::checkbox_sprite_height, kOwner);
    DecodeMember(node, "checkmark_sprite_width", rhs,
                 &arelto::UIMenuConfig::checkmark_sprite_width, kOwner);
    DecodeMember(node, "checkmark_sprite_height", rhs,
                 &arelto::UIMenuConfig::checkmark_sprite_height, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::UICardConfig> {
  static Node encode(const arelto::UICardConfig& rhs) {
    Node node;
    node["level_up_card_width"] = rhs.level_up_card_width;
    node["level_up_card_height"] = rhs.level_up_card_height;
    node["level_up_card_gap"] = rhs.level_up_card_gap;
    node["level_up_icon_offset_y"] = rhs.level_up_icon_offset_y;
    node["level_up_icon_size"] = rhs.level_up_icon_size;
    node["level_up_name_offset_y"] = rhs.level_up_name_offset_y;
    node["level_up_name_offset_x"] = rhs.level_up_name_offset_x;
    node["level_up_name_label_height"] = rhs.level_up_desc_label_height;
    node["level_up_desc_offset_y"] = rhs.level_up_desc_offset_y;
    node["level_up_desc_offset_x"] = rhs.level_up_desc_offset_x;
    node["level_up_desc_label_height"] = rhs.level_up_desc_label_height;
    node["level_up_stats_offset_y"] = rhs.level_up_stats_offset_y;
    node["level_up_stats_offset_x"] = rhs.level_up_stats_offset_x;
    node["level_up_stats_label_height"] = rhs.level_up_stats_label_height;
    node["level_up_row_stride"] = rhs.level_up_row_stride;
    node["level_up_button_offset_y"] = rhs.level_up_button_offset_y;
    node["level_up_button_width"] = rhs.level_up_button_width;
    node["level_up_button_height"] = rhs.level_up_button_height;
    node["item_icon_size"] = rhs.item_icon_size;
    node["item_card_width"] = rhs.item_card_width;
    node["item_card_height"] = rhs.item_card_height;
    node["item_card_gap"] = rhs.item_card_gap;
    node["item_card_icon_offset_y"] = rhs.item_card_icon_offset_y;
    node["item_card_icon_size"] = rhs.item_card_icon_size;
    node["item_card_name_offset_y"] = rhs.item_card_name_offset_y;
    node["item_card_name_offset_x"] = rhs.item_card_name_offset_x;
    node["item_card_desc_offset_y"] = rhs.item_card_desc_offset_y;
    node["item_card_desc_offset_x"] = rhs.item_card_desc_offset_x;
    node["item_card_stats_offset_y"] = rhs.item_card_stats_offset_y;
    node["item_card_stats_offset_x"] = rhs.item_card_stats_offset_x;
    node["item_card_row_stride"] = rhs.item_card_row_stride;
    node["item_card_button_offset_y"] = rhs.item_card_button_offset_y;
    node["item_card_button_width"] = rhs.item_card_button_width;
    node["item_card_button_height"] = rhs.item_card_button_height;
    return node;
  }

  static bool decode(const Node& node, arelto::UICardConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "ui.cards";

    DecodeMember(node, "level_up_card_width", rhs,
                 &arelto::UICardConfig::level_up_card_width, kOwner);
    DecodeMember(node, "level_up_card_height", rhs,
                 &arelto::UICardConfig::level_up_card_height, kOwner);
    DecodeMember(node, "level_up_card_gap", rhs,
                 &arelto::UICardConfig::level_up_card_gap, kOwner);
    DecodeMember(node, "level_up_icon_offset_y", rhs,
                 &arelto::UICardConfig::level_up_icon_offset_y, kOwner);
    DecodeMember(node, "level_up_icon_size", rhs,
                 &arelto::UICardConfig::level_up_icon_size, kOwner);
    DecodeMember(node, "level_up_name_offset_y", rhs,
                 &arelto::UICardConfig::level_up_name_offset_y, kOwner);
    DecodeMember(node, "level_up_name_offset_x", rhs,
                 &arelto::UICardConfig::level_up_name_offset_x, kOwner);
    DecodeMember(node, "level_up_name_label_height", rhs,
                 &arelto::UICardConfig::level_up_name_label_height, kOwner);
    DecodeMember(node, "level_up_desc_offset_y", rhs,
                 &arelto::UICardConfig::level_up_desc_offset_y, kOwner);
    DecodeMember(node, "level_up_desc_offset_x", rhs,
                 &arelto::UICardConfig::level_up_desc_offset_x, kOwner);
    DecodeMember(node, "level_up_desc_label_height", rhs,
                 &arelto::UICardConfig::level_up_desc_label_height, kOwner);
    DecodeMember(node, "level_up_stats_offset_y", rhs,
                 &arelto::UICardConfig::level_up_stats_offset_y, kOwner);
    DecodeMember(node, "level_up_stats_offset_x", rhs,
                 &arelto::UICardConfig::level_up_stats_offset_x, kOwner);
    DecodeMember(node, "level_up_stats_label_height", rhs,
                 &arelto::UICardConfig::level_up_stats_label_height, kOwner);
    DecodeMember(node, "level_up_row_stride", rhs,
                 &arelto::UICardConfig::level_up_row_stride, kOwner);
    DecodeMember(node, "level_up_button_offset_y", rhs,
                 &arelto::UICardConfig::level_up_button_offset_y, kOwner);
    DecodeMember(node, "level_up_button_width", rhs,
                 &arelto::UICardConfig::level_up_button_width, kOwner);
    DecodeMember(node, "level_up_button_height", rhs,
                 &arelto::UICardConfig::level_up_button_height, kOwner);
    DecodeMember(node, "item_icon_size", rhs,
                 &arelto::UICardConfig::item_icon_size, kOwner);
    DecodeMember(node, "item_card_width", rhs,
                 &arelto::UICardConfig::item_card_width, kOwner);
    DecodeMember(node, "item_card_height", rhs,
                 &arelto::UICardConfig::item_card_height, kOwner);
    DecodeMember(node, "item_card_gap", rhs,
                 &arelto::UICardConfig::item_card_gap, kOwner);
    DecodeMember(node, "item_card_icon_offset_y", rhs,
                 &arelto::UICardConfig::item_card_icon_offset_y, kOwner);
    DecodeMember(node, "item_card_icon_size", rhs,
                 &arelto::UICardConfig::item_card_icon_size, kOwner);
    DecodeMember(node, "item_card_name_offset_y", rhs,
                 &arelto::UICardConfig::item_card_name_offset_y, kOwner);
    DecodeMember(node, "item_card_name_offset_x", rhs,
                 &arelto::UICardConfig::item_card_name_offset_x, kOwner);
    DecodeMember(node, "item_card_desc_offset_y", rhs,
                 &arelto::UICardConfig::item_card_desc_offset_y, kOwner);
    DecodeMember(node, "item_card_desc_offset_x", rhs,
                 &arelto::UICardConfig::item_card_desc_offset_x, kOwner);
    DecodeMember(node, "item_card_stats_offset_y", rhs,
                 &arelto::UICardConfig::item_card_stats_offset_y, kOwner);
    DecodeMember(node, "item_card_stats_offset_x", rhs,
                 &arelto::UICardConfig::item_card_stats_offset_x, kOwner);
    DecodeMember(node, "item_card_row_stride", rhs,
                 &arelto::UICardConfig::item_card_row_stride, kOwner);
    DecodeMember(node, "item_card_button_offset_y", rhs,
                 &arelto::UICardConfig::item_card_button_offset_y, kOwner);
    DecodeMember(node, "item_card_button_width", rhs,
                 &arelto::UICardConfig::item_card_button_width, kOwner);
    DecodeMember(node, "item_card_button_height", rhs,
                 &arelto::UICardConfig::item_card_button_height, kOwner);
    return true;
  }
};

template <>
struct convert<arelto::UIInventoryConfig> {
  static Node encode(const arelto::UIInventoryConfig& rhs) {
    Node node;
    node["inventory_bar_y"] = rhs.inventory_bar_y;
    node["inventory_icon_size"] = rhs.inventory_icon_size;
    node["inventory_widget_height"] = rhs.inventory_widget_height;
    node["inventory_label_width"] = rhs.inventory_label_width;
    node["inventory_item_gap"] = rhs.inventory_item_gap;
    node["inventory_multiplier_size"] = rhs.inventory_multiplier_size;
    node["inventory_multiplier_margin"] = rhs.inventory_multiplier_margin;
    node["inventory_container_padding"] = rhs.inventory_container_padding;
    node["inventory_background_alpha"] = rhs.inventory_background_alpha;
    return node;
  }

  static bool decode(const Node& node, arelto::UIInventoryConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    constexpr const char* kOwner = "ui.inventory";

    DecodeMember(node, "inventory_bar_y", rhs,
                 &arelto::UIInventoryConfig::inventory_bar_y, kOwner);
    DecodeMember(node, "inventory_icon_size", rhs,
                 &arelto::UIInventoryConfig::inventory_icon_size, kOwner);
    DecodeMember(node, "inventory_widget_height", rhs,
                 &arelto::UIInventoryConfig::inventory_widget_height, kOwner);
    DecodeMember(node, "inventory_label_width", rhs,
                 &arelto::UIInventoryConfig::inventory_label_width, kOwner);
    DecodeMember(node, "inventory_item_gap", rhs,
                 &arelto::UIInventoryConfig::inventory_item_gap, kOwner);
    DecodeMember(node, "inventory_multiplier_size", rhs,
                 &arelto::UIInventoryConfig::inventory_multiplier_size, kOwner);
    DecodeMember(node, "inventory_multiplier_margin", rhs,
                 &arelto::UIInventoryConfig::inventory_multiplier_margin,
                 kOwner);
    DecodeMember(node, "inventory_container_padding", rhs,
                 &arelto::UIInventoryConfig::inventory_container_padding,
                 kOwner);
    DecodeMember(node, "inventory_background_alpha", rhs,
                 &arelto::UIInventoryConfig::inventory_background_alpha,
                 kOwner);
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
