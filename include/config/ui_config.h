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
  int font_size_small = 14;
  int font_size_medium = 26;
  int font_size_large = 40;
  int font_size_huge = 72;
};

struct UIHudConfig {
  int hud_padding = 50;
  int hud_bar_spacing = 8;
  int bar_text_offset_x = 50;
  int bar_text_offset_y = 0;
  int level_group_offset_y = 55;
  int timer_text_gap = 10;
  int level_text_gap = -2;

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
  int health_bar_text_rel_offset_x = 100;
  int health_bar_text_rel_offset_y = 32;
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
  int exp_bar_text_rel_offset_x = 100;
  int exp_bar_text_rel_offset_y = 32;
  int exp_bar_text_char_width = 20;
  int exp_bar_text_char_height = 25;

  int level_icon_sprite_offset_x = 0;
  int level_icon_sprite_offset_y = 0;
  int level_icon_sprite_width = 70;
  int level_icon_sprite_height = 74;
  int level_text_char_width = 50;
  int level_text_char_height = 72;
  int level_up_icon_margin = -10;
  int level_up_text_margin = -10;
};

struct UIMenuConfig {
  int menu_content_padding = 100;
  int menu_item_spacing = 25;
  int menu_button_gap = 20;
  int menu_bottom_padding = 60;

  int generic_button_texture_width = 300;
  int generic_button_texture_height = 160;

  int begin_button_texture_width = 638;
  int begin_button_texture_height = 540;
  int begin_button_width = 450;
  int begin_button_height = 175;
  int begin_button_y = 2 * (kWindowHeight - begin_button_height) / 7;

  int settings_menu_width = 450;
  int settings_menu_height = 750;
  int settings_menu_background_sprite_width = 900;
  int settings_menu_background_sprite_height = 1000;
  int settings_menu_button_width = 150;
  int settings_menu_button_height = 50;
  int settings_menu_volume_slider_width = 300;
  int settings_menu_volume_slider_height = 30;
  int volume_slider_fill_offset_x = 15;
  int volume_slider_fill_offset_y = 5;
  int volume_slider_fill_width = 275;
  int volume_slider_fill_height = 20;

  int quit_menu_width = 550;
  int quit_menu_height = 300;

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
  int level_up_card_width = 400;
  int level_up_card_height = 600;
  int level_up_card_gap = 100;
  int level_up_icon_offset_y = 120;
  int level_up_icon_size = 80;
  int level_up_name_offset_y = 220;
  int level_up_name_offset_x = 70;
  int level_up_desc_offset_y = 300;
  int level_up_desc_offset_x = 70;
  int level_up_stats_offset_y = 350;
  int level_up_stats_offset_x = 70;
  int level_up_row_stride = 55;
  int level_up_button_offset_y = 440;
  int level_up_button_width = 200;
  int level_up_button_height = 50;

  int item_icon_size = 300;
  int item_card_width = 650;
  int item_card_height = 1000;
  int item_card_gap = 150;
  int item_card_icon_offset_y = 160;
  int item_card_icon_size = 80;
  int item_card_name_offset_y = 475;
  int item_card_name_offset_x = 100;
  int item_card_desc_offset_y = 600;
  int item_card_desc_offset_x = 70;
  int item_card_stats_offset_y = 648;
  int item_card_stats_offset_x = 70;
  int item_card_row_stride = 100;
  int item_card_button_offset_y = 800;
  int item_card_button_width = 200;
  int item_card_button_height = 50;
};

struct UIInventoryConfig {
  int inventory_bar_y = 50;
  int inventory_icon_size = 60;
  int inventory_widget_height = 60;
  int inventory_label_width = 20;
  int inventory_item_gap = 15;
  int inventory_multiplier_size = 16;
  int inventory_multiplier_margin = 0;
  int inventory_container_padding = 20;
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
