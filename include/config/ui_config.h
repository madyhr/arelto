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
  int font_size_small = 18;
  int font_size_medium = 26;
  int font_size_large = 40;
  int font_size_huge = 72;
};

struct UIHudConfig {
  float hud_padding = 50.0f;
  float hud_bar_spacing = 8.0f;
  float bar_text_offset_x = 50.0f;
  float bar_text_offset_y = 0.0f;
  float level_group_offset_y = 55.0f;
  float timer_text_gap = 10.0f;
  float level_text_gap = -2.0f;

  int digit_sprite_width = 30;
  int digit_sprite_height = 50;

  int health_bar_container_sprite_offset_x = 0;
  int health_bar_container_sprite_offset_y = 0;
  int health_bar_container_sprite_width = 404;
  int health_bar_container_sprite_height = 92;
  float health_bar_rel_offset_x = 80.0f;
  float health_bar_rel_offset_y = 32.0f;
  int health_bar_sprite_offset_x = 0;
  int health_bar_sprite_offset_y = 128;
  int health_bar_sprite_width = 299;
  int health_bar_sprite_height = 28;
  float health_bar_text_rel_offset_x = 100.0f;
  float health_bar_text_rel_offset_y = 32.0f;
  int health_bar_text_char_width = 20;
  int health_bar_text_char_height = 25;

  int timer_hourglass_sprite_width = 50;
  int timer_hourglass_sprite_height = 72;
  int timer_text_char_width = 50;
  int timer_text_char_height = 72;

  int game_over_sprite_width = 610;
  int game_over_sprite_height = 88;

  int exp_bar_container_sprite_offset_x = 0;
  int exp_bar_container_sprite_offset_y = 0;
  int exp_bar_container_sprite_width = 404;
  int exp_bar_container_sprite_height = 92;
  float exp_bar_rel_offset_x = 80.0f;
  float exp_bar_rel_offset_y = 30.0f;
  int exp_bar_sprite_offset_x = 0;
  int exp_bar_sprite_offset_y = 128;
  int exp_bar_sprite_width = 299;
  int exp_bar_sprite_height = 28;
  float exp_bar_text_rel_offset_x = 100.0f;
  float exp_bar_text_rel_offset_y = 32.0f;
  int exp_bar_text_char_width = 20;
  int exp_bar_text_char_height = 25;

  int level_icon_sprite_offset_x = 0;
  int level_icon_sprite_offset_y = 0;
  int level_icon_sprite_width = 70;
  int level_icon_sprite_height = 74;
  int level_text_char_width = 50;
  int level_text_char_height = 72;
  float level_up_icon_margin = -10.0f;
  float level_up_text_margin = -10.0f;
};

struct UIMenuConfig {
  float menu_content_padding = 100.0f;
  float menu_item_spacing = 25.0f;
  float menu_button_gap = 20.0f;
  float menu_bottom_padding = 60.0f;

  int generic_button_texture_width = 300;
  int generic_button_texture_height = 160;

  int begin_button_texture_width = 638;
  int begin_button_texture_height = 540;
  float begin_button_width = 450.0f;
  float begin_button_height = 175.0f;
  float begin_button_y = 2.0f * (kWindowHeight - begin_button_height) / 7.0f;

  float settings_menu_width = 450.0f;
  float settings_menu_height = 750.0f;
  int settings_menu_background_sprite_width = 900;
  int settings_menu_background_sprite_height = 1000;
  float settings_menu_button_width = 150.0f;
  float settings_menu_button_height = 50.0f;
  float settings_menu_volume_slider_width = 300.0f;
  float settings_menu_volume_slider_height = 30.0f;
  float volume_slider_fill_offset_x = 15.0f;
  float volume_slider_fill_offset_y = 5.0f;
  int volume_slider_fill_width = 275;
  int volume_slider_fill_height = 20;

  float quit_menu_width = 550.0f;
  float quit_menu_height = 300.0f;

  int slider_container_sprite_offset_x = 0;
  int slider_container_sprite_offset_y = 0;
  int slider_container_sprite_width = 882;
  int slider_container_sprite_height = 48;
  int slider_bar_sprite_offset_x = 0;
  int slider_bar_sprite_offset_y = 48;
  int slider_bar_sprite_width = 806;
  int slider_bar_sprite_height = 29;

  int checkbox_sprite_width = 263;
  int checkbox_sprite_height = 526;
  int checkmark_sprite_width = 193;
  int checkmark_sprite_height = 164;
};

struct UICardConfig {
  float level_up_card_width = 400.0f;
  float level_up_card_height = 600.0f;
  float level_up_card_gap = 100.0f;
  float level_up_icon_offset_y = 120.0f;
  float level_up_icon_size = 80.0f;
  float level_up_name_offset_y = 220.0f;
  float level_up_name_offset_x = 70.0f;
  float level_up_name_label_height = 96.0f;
  float level_up_desc_offset_y = 300.0f;
  float level_up_desc_offset_x = 70.0f;
  float level_up_desc_label_height = 15.0f;
  float level_up_stats_offset_y = 350.0f;
  float level_up_stats_offset_x = 70.0f;
  float level_up_stats_label_height = 15.0f;
  float level_up_row_stride = 55.0f;
  float level_up_button_offset_y = 440.0f;
  float level_up_button_width = 200.0f;
  float level_up_button_height = 50.0f;

  float item_icon_size = 300.0f;
  float item_card_width = 650.0f;
  float item_card_height = 1000.0f;
  float item_card_gap = 150.0f;
  float item_card_icon_offset_y = 160.0f;
  float item_card_name_offset_x = 100.0f;
  float item_card_name_offset_y = 475.0f;
  float item_card_flavor_text_offset_x = 100.0f;
  float item_card_flavor_text_offset_y = 525.0f;
  float item_card_desc_offset_x = 70.0f;
  float item_card_desc_offset_y = 600.0f;
  float item_card_stats_offset_x = 70.0f;
  float item_card_stats_offset_y = 648.0f;
  float item_card_row_stride = 100.0f;
  float item_card_button_offset_y = 800.0f;
  float item_card_button_width = 200.0f;
  float item_card_button_height = 50.0f;
};

struct UIInventoryConfig {
  float inventory_bar_y = 50.0f;
  float inventory_icon_size = 60.0f;
  float inventory_widget_height = 60.0f;
  float inventory_label_width = 20.0f;
  float inventory_item_gap = 15.0f;
  float inventory_multiplier_size = 16.0f;
  float inventory_multiplier_margin = 0.0f;
  float inventory_container_padding = 20.0f;
  int inventory_background_alpha = 64;
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
