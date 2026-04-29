// src/ui_manager.cpp
#include "ui_manager.h"
#include <functional>
#include <string>
#include "constants/chest.h"
#include "constants/progression_manager.h"
#include "constants/projectile.h"
#include "entity.h"
#include "event_manager.h"
#include "scene.h"
#include "ui/containers.h"
#include "ui/widget.h"
#include "ui/widgets.h"

namespace arelto {

void UIManager::SetupUI(const UIResources& resources, const UIConfig& config,
                        EventManager& event_manager) {
  resources_ = &resources;
  ui_config_ = config;

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
  BuildChestOpeningScreen();
  BuildItemInventory();

  SetupUIEventSubscriptions(event_manager);

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

UIWidget* UIManager::GetItemMenuRoot() {
  return root_widget_ ? root_widget_->FindWidget("item_menu") : nullptr;
}

// =============================================================================
// BuildHUD: Health Bar, Exp Bar, Level Indicator, Timer
// =============================================================================

void UIManager::BuildHUD() {
  // --- Top-left group: Timer + Level Indicator ---
  auto top_left = std::make_shared<VBox>();
  top_left->SetId("hud_top_left");
  top_left->SetAnchor(AnchorType::TopLeft);
  top_left->SetPosition(ui_config_.hud.hud_padding, ui_config_.hud.hud_padding);
  top_left->SetSize(200, 200);
  top_left->SetSpacing(ui_config_.hud.level_group_offset_y);

  auto timer_row = std::make_shared<HBox>();
  timer_row->SetId("timer_row");
  timer_row->SetSize(200, ui_config_.hud.timer_hourglass_sprite_height);
  timer_row->SetSpacing(ui_config_.hud.timer_text_gap);

  auto timer_icon = std::make_shared<UIImage>();
  timer_icon->SetId("timer_icon");
  timer_icon->SetSize(ui_config_.hud.timer_hourglass_sprite_width,
                      ui_config_.hud.timer_hourglass_sprite_height);
  timer_icon->SetTexture(resources_->timer_hourglass_texture);
  timer_icon->SetSrcRect({0, 0, ui_config_.hud.timer_hourglass_sprite_width,
                          ui_config_.hud.timer_hourglass_sprite_height});
  timer_row->AddChild(timer_icon);

  auto timer_text = std::make_shared<UILabel>();
  timer_text->SetId("timer_text");
  timer_text->SetSize(ui_config_.hud.timer_text_char_width * 3,
                      ui_config_.hud.timer_text_char_height);
  timer_text->SetUseDigitFont(true);
  timer_text->SetDigitSpriteSize(ui_config_.hud.digit_sprite_width,
                                 ui_config_.hud.digit_sprite_height);
  timer_text->SetCharSize(ui_config_.hud.timer_text_char_width,
                          ui_config_.hud.timer_text_char_height);
  timer_row->AddChild(timer_text);

  top_left->AddChild(timer_row);

  auto level_row = std::make_shared<HBox>();
  level_row->SetId("level_row");
  level_row->SetSize(200, ui_config_.hud.level_icon_sprite_height);

  auto level_icon = std::make_shared<UIImage>();
  level_icon->SetId("level_icon");
  level_icon->SetSize(ui_config_.hud.level_icon_sprite_width,
                      ui_config_.hud.level_icon_sprite_height);
  level_icon->SetTexture(resources_->level_indicator_texture);
  level_icon->SetSrcRect({ui_config_.hud.level_icon_sprite_offset_x,
                          ui_config_.hud.level_icon_sprite_offset_y,
                          ui_config_.hud.level_icon_sprite_width,
                          ui_config_.hud.level_icon_sprite_height});
  level_icon->SetMargin(ui_config_.hud.level_up_icon_margin);
  level_row->AddChild(level_icon);

  auto level_text = std::make_shared<UILabel>();
  level_text->SetId("level_text");
  level_text->SetSize(ui_config_.hud.level_text_char_width * 3,
                      ui_config_.hud.level_text_char_height);
  level_text->SetUseDigitFont(true);
  level_text->SetDigitSpriteSize(ui_config_.hud.digit_sprite_width,
                                 ui_config_.hud.digit_sprite_height);
  level_text->SetCharSize(ui_config_.hud.level_text_char_width,
                          ui_config_.hud.level_text_char_height);
  level_row->AddChild(level_text);
  level_text->SetMargin(ui_config_.hud.level_up_text_margin);

  top_left->AddChild(level_row);
  root_widget_->AddChild(top_left);

  // --- Bottom-left group: Health Bar + Exp Bar ---
  auto bottom_left = std::make_shared<VBox>();
  bottom_left->SetId("hud_bottom_left");
  bottom_left->SetAnchor(AnchorType::BottomLeft);
  bottom_left->SetPosition(ui_config_.hud.hud_padding,
                           -ui_config_.hud.hud_padding);
  bottom_left->SetSize(ui_config_.hud.health_bar_container_sprite_width,
                       ui_config_.hud.health_bar_container_sprite_height * 2 +
                           ui_config_.hud.hud_bar_spacing);
  bottom_left->SetSpacing(ui_config_.hud.hud_bar_spacing);

  auto health_bar = std::make_shared<UIProgressBar>();
  health_bar->SetId("health_bar");
  health_bar->SetSize(ui_config_.hud.health_bar_container_sprite_width,
                      ui_config_.hud.health_bar_container_sprite_height);
  health_bar->SetContainerTexture(resources_->health_bar_texture);
  health_bar->SetContainerSrcRect(
      {ui_config_.hud.health_bar_container_sprite_offset_x,
       ui_config_.hud.health_bar_container_sprite_offset_y,
       ui_config_.hud.health_bar_container_sprite_width,
       ui_config_.hud.health_bar_container_sprite_height});
  health_bar->SetFillTexture(resources_->health_bar_texture);
  health_bar->SetFillSrcRect({ui_config_.hud.health_bar_sprite_offset_x,
                              ui_config_.hud.health_bar_sprite_offset_y,
                              ui_config_.hud.health_bar_sprite_width,
                              ui_config_.hud.health_bar_sprite_height});
  health_bar->SetFillOffset(ui_config_.hud.health_bar_rel_offset_x,
                            ui_config_.hud.health_bar_rel_offset_y);
  health_bar->SetMaxFillSize(ui_config_.hud.health_bar_sprite_width,
                             ui_config_.hud.health_bar_sprite_height);
  health_bar->SetPercent(1.0f);

  auto health_text = std::make_shared<UILabel>();
  health_text->SetId("health_text");
  health_text->SetPosition(ui_config_.hud.health_bar_text_rel_offset_x,
                           ui_config_.hud.health_bar_text_rel_offset_y);
  health_text->SetSize(ui_config_.hud.digit_sprite_width,
                       ui_config_.hud.digit_sprite_height);
  health_text->SetUseDigitFont(true);
  health_text->SetDigitSpriteSize(ui_config_.hud.digit_sprite_width,
                                  ui_config_.hud.digit_sprite_height);
  health_text->SetCharSize(ui_config_.hud.health_bar_text_char_width,
                           ui_config_.hud.health_bar_text_char_height);
  health_bar->AddChild(health_text);

  bottom_left->AddChild(health_bar);

  auto exp_bar = std::make_shared<UIProgressBar>();
  exp_bar->SetId("exp_bar");
  exp_bar->SetSize(ui_config_.hud.exp_bar_container_sprite_width,
                   ui_config_.hud.exp_bar_container_sprite_height);
  exp_bar->SetContainerTexture(resources_->exp_bar_texture);
  exp_bar->SetContainerSrcRect(
      {ui_config_.hud.exp_bar_container_sprite_offset_x,
       ui_config_.hud.exp_bar_container_sprite_offset_y,
       ui_config_.hud.exp_bar_container_sprite_width,
       ui_config_.hud.exp_bar_container_sprite_height});
  exp_bar->SetFillTexture(resources_->exp_bar_texture);
  exp_bar->SetFillSrcRect({ui_config_.hud.exp_bar_sprite_offset_x,
                           ui_config_.hud.exp_bar_sprite_offset_y,
                           ui_config_.hud.exp_bar_sprite_width,
                           ui_config_.hud.exp_bar_sprite_height});
  exp_bar->SetFillOffset(ui_config_.hud.exp_bar_rel_offset_x,
                         ui_config_.hud.exp_bar_rel_offset_y);
  exp_bar->SetMaxFillSize(ui_config_.hud.exp_bar_sprite_width,
                          ui_config_.hud.exp_bar_sprite_height);
  exp_bar->SetPercent(0.0f);

  auto exp_text = std::make_shared<UILabel>();
  exp_text->SetId("exp_text");
  exp_text->SetPosition(ui_config_.hud.exp_bar_text_rel_offset_x,
                        ui_config_.hud.exp_bar_text_rel_offset_y);
  exp_text->SetSize(ui_config_.hud.digit_sprite_width,
                    ui_config_.hud.digit_sprite_height);
  exp_text->SetUseDigitFont(true);
  exp_text->SetDigitSpriteSize(ui_config_.hud.digit_sprite_width,
                               ui_config_.hud.digit_sprite_height);
  exp_text->SetCharSize(ui_config_.hud.exp_bar_text_char_width,
                        ui_config_.hud.exp_bar_text_char_height);
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
  menu->SetSize(ui_config_.menus.settings_menu_width,
                ui_config_.menus.settings_menu_height);
  menu->SetBackground(resources_->settings_menu_background_texture);
  menu->SetBackgroundSrcRect(
      {0, 0, ui_config_.menus.settings_menu_background_sprite_width,
       ui_config_.menus.settings_menu_background_sprite_height});

  auto content = std::make_shared<VBox>();
  content->SetId("settings_content");
  content->SetSize(ui_config_.menus.settings_menu_width,
                   ui_config_.menus.settings_menu_height);
  content->SetPadding(ui_config_.menus.menu_content_padding);
  content->SetSpacing(ui_config_.menus.menu_item_spacing);

  auto title_spacer = std::make_shared<Spacer>(0, 5);
  content->AddChild(title_spacer);

  auto title = std::make_shared<UILabel>();
  title->SetId("settings_title");
  title->SetSize(ui_config_.menus.settings_menu_width -
                     2 * ui_config_.menus.menu_content_padding,
                 100);
  title->SetText("SETTINGS");
  title->SetFont(resources_->ui_font_huge);
  title->SetCenterWidth(ui_config_.menus.settings_menu_width -
                        2 * ui_config_.menus.menu_content_padding);
  content->AddChild(title);

  auto volume_label = std::make_shared<UILabel>();
  volume_label->SetId("volume_label");
  volume_label->SetSize(ui_config_.menus.settings_menu_width -
                            2 * ui_config_.menus.menu_content_padding,
                        30);
  volume_label->SetText("MUSIC VOLUME");
  volume_label->SetFont(resources_->ui_font_large);
  volume_label->SetCenterWidth(ui_config_.menus.settings_menu_width -
                               2 * ui_config_.menus.menu_content_padding);
  content->AddChild(volume_label);

  auto volume_slider = std::make_shared<UIProgressBar>();
  volume_slider->SetId("volume_slider");
  volume_slider->SetAnchor(AnchorType::TopCenter);
  volume_slider->SetSize(ui_config_.menus.settings_menu_volume_slider_width,
                         ui_config_.menus.settings_menu_volume_slider_height);
  volume_slider->SetContainerTexture(resources_->slider_texture);
  volume_slider->SetContainerSrcRect(
      {ui_config_.menus.slider_container_sprite_offset_x,
       ui_config_.menus.slider_container_sprite_offset_y,
       ui_config_.menus.slider_container_sprite_width,
       ui_config_.menus.slider_container_sprite_height});
  volume_slider->SetFillTexture(resources_->slider_texture);
  volume_slider->SetFillSrcRect({ui_config_.menus.slider_bar_sprite_offset_x,
                                 ui_config_.menus.slider_bar_sprite_offset_y,
                                 ui_config_.menus.slider_bar_sprite_width,
                                 ui_config_.menus.slider_bar_sprite_height});
  volume_slider->SetFillOffset(ui_config_.menus.volume_slider_fill_offset_x,
                               ui_config_.menus.volume_slider_fill_offset_y);
  volume_slider->SetMaxFillSize(ui_config_.menus.volume_slider_fill_width,
                                ui_config_.menus.volume_slider_fill_height);
  volume_slider->SetPercent(1.0f);
  content->AddChild(volume_slider);

  auto mute_row = std::make_shared<HBox>();
  mute_row->SetId("mute_row");
  mute_row->SetSize(ui_config_.menus.settings_menu_width -
                        2 * ui_config_.menus.menu_content_padding,
                    40);
  mute_row->SetSpacing(10);
  mute_row->SetAnchor(AnchorType::TopCenter);

  auto mute_checkbox = std::make_shared<UICheckbox>();
  mute_checkbox->SetId("mute_checkbox");
  mute_checkbox->SetSize(30, 30);
  mute_checkbox->SetBoxTexture(resources_->checkbox_texture);
  mute_checkbox->SetBoxSrcRect({0, 0, ui_config_.menus.checkbox_sprite_width,
                                ui_config_.menus.checkbox_sprite_height / 2});
  mute_checkbox->SetBoxHoverSrcRect(
      {0, ui_config_.menus.checkbox_sprite_height / 2,
       ui_config_.menus.checkbox_sprite_width,
       ui_config_.menus.checkbox_sprite_height / 2});
  mute_checkbox->SetMarkTexture(resources_->checkmark_texture);
  mute_checkbox->SetMarkSrcRect({0, 0, ui_config_.menus.checkmark_sprite_width,
                                 ui_config_.menus.checkmark_sprite_height});
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
  debug_label->SetSize(ui_config_.menus.settings_menu_width -
                           2 * ui_config_.menus.menu_content_padding,
                       40);
  debug_label->SetText("DEBUG");
  debug_label->SetFont(resources_->ui_font_large);
  debug_label->SetCenterWidth(ui_config_.menus.settings_menu_width -
                              2 * ui_config_.menus.menu_content_padding);
  content->AddChild(debug_label);

  auto occupancy_map_row = std::make_shared<HBox>();
  occupancy_map_row->SetId("occupancy_map_row");
  occupancy_map_row->SetSize(ui_config_.menus.settings_menu_width -
                                 2 * ui_config_.menus.menu_content_padding,
                             40);
  occupancy_map_row->SetSpacing(10);
  occupancy_map_row->SetAnchor(AnchorType::TopCenter);

  auto occupancy_map_checkbox = std::make_shared<UICheckbox>();
  occupancy_map_checkbox->SetId("occupancy_map_checkbox");
  occupancy_map_checkbox->SetSize(30, 30);
  occupancy_map_checkbox->SetBoxTexture(resources_->checkbox_texture);
  occupancy_map_checkbox->SetBoxSrcRect(
      {0, 0, ui_config_.menus.checkbox_sprite_width,
       ui_config_.menus.checkbox_sprite_height / 2});
  occupancy_map_checkbox->SetBoxHoverSrcRect(
      {0, ui_config_.menus.checkbox_sprite_height / 2,
       ui_config_.menus.checkbox_sprite_width,
       ui_config_.menus.checkbox_sprite_height / 2});
  occupancy_map_checkbox->SetMarkTexture(resources_->checkmark_texture);
  occupancy_map_checkbox->SetMarkSrcRect(
      {0, 0, ui_config_.menus.checkmark_sprite_width,
       ui_config_.menus.checkmark_sprite_height});
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
  ray_caster_row->SetSize(ui_config_.menus.settings_menu_width -
                              2 * ui_config_.menus.menu_content_padding,
                          40);
  ray_caster_row->SetSpacing(10);
  ray_caster_row->SetAnchor(AnchorType::TopCenter);

  auto ray_caster_checkbox = std::make_shared<UICheckbox>();
  ray_caster_checkbox->SetId("ray_caster_checkbox");
  ray_caster_checkbox->SetSize(30, 30);
  ray_caster_checkbox->SetBoxTexture(resources_->checkbox_texture);
  ray_caster_checkbox->SetBoxSrcRect(
      {0, 0, ui_config_.menus.checkbox_sprite_width,
       ui_config_.menus.checkbox_sprite_height / 2});
  ray_caster_checkbox->SetBoxHoverSrcRect(
      {0, ui_config_.menus.checkbox_sprite_height / 2,
       ui_config_.menus.checkbox_sprite_width,
       ui_config_.menus.checkbox_sprite_height / 2});
  ray_caster_checkbox->SetMarkTexture(resources_->checkmark_texture);
  ray_caster_checkbox->SetMarkSrcRect(
      {0, 0, ui_config_.menus.checkmark_sprite_width,
       ui_config_.menus.checkmark_sprite_height});
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
  button_row->SetPosition(0, -ui_config_.menus.menu_bottom_padding);
  button_row->SetSize(ui_config_.menus.settings_menu_button_width * 2 +
                          ui_config_.menus.menu_button_gap,
                      ui_config_.menus.settings_menu_button_height);
  button_row->SetSpacing(ui_config_.menus.menu_button_gap);

  auto resume_btn = std::make_shared<UIButton>();
  resume_btn->SetId("resume_button");
  resume_btn->SetSize(ui_config_.menus.settings_menu_button_width,
                      ui_config_.menus.settings_menu_button_height);
  resume_btn->SetTexture(resources_->button_texture);
  resume_btn->SetNormalSrcRect(
      {0, 0, ui_config_.menus.generic_button_texture_width,
       ui_config_.menus.generic_button_texture_height / 2});
  resume_btn->SetHoverSrcRect(
      {0, ui_config_.menus.generic_button_texture_height / 2,
       ui_config_.menus.generic_button_texture_width,
       ui_config_.menus.generic_button_texture_height / 2});
  resume_btn->SetLabel("RESUME");
  resume_btn->SetLabelFont(resources_->ui_font_medium);
  button_row->AddChild(resume_btn);

  auto main_menu_btn = std::make_shared<UIButton>();
  main_menu_btn->SetId("main_menu_button");
  main_menu_btn->SetSize(ui_config_.menus.settings_menu_button_width,
                         ui_config_.menus.settings_menu_button_height);
  main_menu_btn->SetTexture(resources_->button_texture);
  main_menu_btn->SetNormalSrcRect(
      {0, 0, ui_config_.menus.generic_button_texture_width,
       ui_config_.menus.generic_button_texture_height / 2});
  main_menu_btn->SetHoverSrcRect(
      {0, ui_config_.menus.generic_button_texture_height / 2,
       ui_config_.menus.generic_button_texture_width,
       ui_config_.menus.generic_button_texture_height / 2});
  main_menu_btn->SetLabel("MAIN MENU");
  main_menu_btn->SetLabelFont(resources_->ui_font_medium);
  button_row->AddChild(main_menu_btn);

  menu->AddChild(button_row);

  overlay->AddChild(menu);
  root_widget_->AddChild(overlay);
}

// =============================================================================
// BuildLevelUpMenu: dynamically create card widgets from upgrade options
// =============================================================================

void UIManager::BuildLevelUpMenu(const UpgradeOptions& options) {
  root_widget_->RemoveChild("level_up_menu");

  auto overlay = std::make_shared<Panel>();
  overlay->SetId("level_up_menu");
  overlay->SetSize(kWindowWidth, kWindowHeight);
  overlay->SetBackgroundColor(WithOpacity(kColorBlack, 128));
  overlay->SetVisible(false);

  auto card_row = std::make_shared<HBox>();
  card_row->SetId("level_up_cards");
  card_row->SetAnchor(AnchorType::Center);
  int total_width =
      kNumSpellUpgradeOptions * ui_config_.cards.level_up_card_width +
      (kNumSpellUpgradeOptions - 1) * ui_config_.cards.level_up_card_gap;
  card_row->SetSize(static_cast<float>(total_width),
                    ui_config_.cards.level_up_card_height);
  card_row->SetSpacing(ui_config_.cards.level_up_card_gap);

  for (size_t i = 0; i < options.size(); ++i) {
    BuildLevelUpCard(card_row.get(), static_cast<int>(i),
                     static_cast<SpellStatUpgrade&>(*options[i]));
  }

  overlay->AddChild(card_row);
  root_widget_->AddChild(overlay);
  root_widget_->ComputeLayout(0, 0, kWindowWidth, kWindowHeight);
}

void UIManager::BuildLevelUpCard(UIWidget* parent, int index,
                                 const SpellStatUpgrade& upgrade) {
  std::string card_id = "level_up_card_" + std::to_string(index);

  auto card = std::make_shared<Panel>();
  card->SetId(card_id);
  card->SetSize(ui_config_.cards.level_up_card_width,
                ui_config_.cards.level_up_card_height);
  card->SetBackground(resources_->level_up_option_card_texture);
  card->SetBackgroundSrcRect({0, 0, 0, 0});  // full texture

  int spell_id = upgrade.GetSpellID();
  if (spell_id >= 0 &&
      spell_id < static_cast<int>(resources_->projectile_textures.size())) {
    auto icon = std::make_shared<UIImage>();
    icon->SetId(card_id + "_icon");
    icon->SetAnchor(AnchorType::TopCenter);
    icon->SetPosition(0, ui_config_.cards.level_up_icon_offset_y);
    icon->SetSize(ui_config_.cards.level_up_icon_size,
                  ui_config_.cards.level_up_icon_size);
    icon->SetTexture(resources_->projectile_textures[spell_id]);
    icon->SetSrcRect({0, 0, kFireballSpriteWidth, kFireballSpriteHeight});
    card->AddChild(icon);
  }

  auto name_label = std::make_shared<UILabel>();
  name_label->SetId(card_id + "_name");
  name_label->SetPosition(ui_config_.cards.level_up_name_offset_x,
                          ui_config_.cards.level_up_name_offset_y);
  name_label->SetSize(ui_config_.cards.level_up_card_width -
                          2 * ui_config_.cards.level_up_name_offset_x,
                      96);
  name_label->SetText(upgrade.GetName());
  name_label->SetFont(resources_->ui_font_large);
  name_label->SetColor({255, 255, 255, 255});
  name_label->SetCenterWidth(ui_config_.cards.level_up_card_width -
                             2 * ui_config_.cards.level_up_name_offset_x);
  name_label->SetWrapWidth(ui_config_.cards.level_up_card_width -
                           2 * ui_config_.cards.level_up_name_offset_x);
  card->AddChild(name_label);

  std::vector<UpgradeDisplayRow> display_rows = upgrade.GetDisplayRows();
  for (size_t row_index = 0; row_index < display_rows.size(); ++row_index) {
    const UpgradeDisplayRow& row = display_rows[row_index];
    int row_desc_y =
        ui_config_.cards.level_up_desc_offset_y +
        static_cast<int>(row_index) * ui_config_.cards.level_up_row_stride;
    int row_stats_y =
        ui_config_.cards.level_up_stats_offset_y +
        static_cast<int>(row_index) * ui_config_.cards.level_up_row_stride;

    auto desc_label = std::make_shared<UILabel>();
    desc_label->SetId(card_id + "_desc_" + std::to_string(row_index));
    desc_label->SetPosition(ui_config_.cards.level_up_desc_offset_x,
                            static_cast<float>(row_desc_y));
    desc_label->SetSize(ui_config_.cards.level_up_card_width -
                            2 * ui_config_.cards.level_up_desc_offset_x,
                        25);
    desc_label->SetText(row.description);
    desc_label->SetFont(resources_->ui_font_medium);
    desc_label->SetColor({180, 180, 180, 255});
    desc_label->SetCenterWidth(ui_config_.cards.level_up_card_width -
                               2 * ui_config_.cards.level_up_desc_offset_x);
    card->AddChild(desc_label);

    if (row.old_value.empty() && row.new_value.empty()) {
      continue;
    }
    std::string stats_str = row.old_value + " -> " + row.new_value;
    auto stats_label = std::make_shared<UILabel>();
    stats_label->SetId(card_id + "_stats_" + std::to_string(row_index));
    stats_label->SetPosition(ui_config_.cards.level_up_stats_offset_x,
                             static_cast<float>(row_stats_y));
    stats_label->SetSize(ui_config_.cards.level_up_card_width -
                             2 * ui_config_.cards.level_up_stats_offset_x,
                         25);
    stats_label->SetText(stats_str);
    stats_label->SetFont(resources_->ui_font_medium);
    // NOTE: The assumption here is that all stat upgrades are positive (e.g. "Damage: 10 -> 12").
    stats_label->SetColor(ui_config_.colors.positive_green);
    stats_label->SetCenterWidth(ui_config_.cards.level_up_card_width -
                                2 * ui_config_.cards.level_up_stats_offset_x);
    card->AddChild(stats_label);
  }

  std::string btn_id = "select_button_" + std::to_string(index);
  auto select_btn = std::make_shared<UIButton>();
  select_btn->SetId(btn_id);
  select_btn->SetAnchor(AnchorType::TopCenter);
  select_btn->SetPosition(0, ui_config_.cards.level_up_button_offset_y);
  select_btn->SetSize(ui_config_.cards.level_up_button_width,
                      ui_config_.cards.level_up_button_height);
  select_btn->SetTexture(resources_->button_texture);
  select_btn->SetNormalSrcRect(
      {0, 0, ui_config_.menus.generic_button_texture_width,
       ui_config_.menus.generic_button_texture_height / 2});
  select_btn->SetHoverSrcRect(
      {0, ui_config_.menus.generic_button_texture_height / 2,
       ui_config_.menus.generic_button_texture_width,
       ui_config_.menus.generic_button_texture_height / 2});
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
// ======================================================================================
// Event subscriptions: Add subscriptions to game events that should trigger UI updates.
// ======================================================================================

void UIManager::SetupUIEventSubscriptions(EventManager& event_manager) {
  event_manager.Subscribe<SceneResetEvent>([this](const SceneResetEvent&,
                                                  EventContext& event_context) {
    UpdateExpBar(
        event_context.scene.player.stats_.exp_points,
        event_context.scene.player.stats_.exp_points_required.GetValueCeil());
    auto* level_text = GetWidget<UILabel>("level_text");
    if (level_text) {
      level_text->SetText(
          std::to_string(event_context.scene.player.stats_.level));
    }
    UpdateHealthBar(
        event_context.scene.player.stats_.health,
        event_context.scene.player.stats_.max_health.GetValueCeil());
    UpdateItemInventory(event_context.scene.player.inventory_);
  });
  event_manager.Subscribe<ExpGemCollectedEvent>(
      [this](const ExpGemCollectedEvent&, EventContext& event_context) {
        UpdateExpBar(event_context.scene.player.stats_.exp_points,
                     event_context.scene.player.stats_.exp_points_required
                         .GetValueCeil());
      });

  event_manager.Subscribe<PlayerLevelUpEvent>([this](
                                                  const PlayerLevelUpEvent&,
                                                  EventContext& event_context) {
    UpdateExpBar(
        event_context.scene.player.stats_.exp_points,
        event_context.scene.player.stats_.exp_points_required.GetValueCeil());

    auto* level_text = GetWidget<UILabel>("level_text");
    if (level_text) {
      level_text->SetText(
          std::to_string(event_context.scene.player.stats_.level));
    }
  });

  event_manager.Subscribe<PlayerDamagedEvent>(
      [this](const PlayerDamagedEvent&, EventContext& event_context) {
        UpdateHealthBar(
            event_context.scene.player.stats_.health,
            event_context.scene.player.stats_.max_health.GetValueCeil());
      });

  event_manager.Subscribe<PlayerHealedEvent>(
      [this](const PlayerHealedEvent&, EventContext& event_context) {
        UpdateHealthBar(
            event_context.scene.player.stats_.health,
            event_context.scene.player.stats_.max_health.GetValueCeil());
      });

  event_manager.Subscribe<PlayerClaimedItemEvent>(
      [this](const PlayerClaimedItemEvent&, EventContext& event_context) {
        UpdateItemInventory(event_context.scene.player.inventory_);
      });
}

void UIManager::UpdateTimer(float time) {
  auto* timer_text = GetWidget<UILabel>("timer_text");
  if (timer_text) {
    timer_text->SetText(std::to_string(static_cast<int>(time)));
  }
}

void UIManager::UpdateExpBar(int current_exp_points, int exp_points_required) {
  auto* exp_bar = GetWidget<UIProgressBar>("exp_bar");
  if (exp_bar) {
    float max_exp = static_cast<float>(exp_points_required);
    float percent = static_cast<float>(current_exp_points) / max_exp;
    exp_bar->SetPercent(percent);
  }

  auto* exp_text = GetWidget<UILabel>("exp_text");
  if (exp_text) {
    exp_text->SetText(std::to_string(current_exp_points) + "/" +
                      std::to_string(exp_points_required));
  }
};

void UIManager::UpdateHealthBar(int current_health_points,
                                int max_health_points) {
  auto* health_bar = GetWidget<UIProgressBar>("health_bar");
  if (health_bar) {
    float max_hp = static_cast<float>(max_health_points);
    float percent = static_cast<float>(current_health_points) / max_hp;
    health_bar->SetPercent(percent);
  }

  auto* health_text = GetWidget<UILabel>("health_text");
  if (health_text) {
    health_text->SetText(std::to_string(current_health_points) + "/" +
                         std::to_string(max_health_points));
  }
}

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
  // A negative sign is used for the Y coordinate of the begin button as it is_muted
  // anchored to the bottom, so we want to displace it upwards (reduce Y).
  begin_btn->SetPosition(0,
                         -static_cast<float>(ui_config_.menus.begin_button_y));
  begin_btn->SetSize(ui_config_.menus.begin_button_width,
                     ui_config_.menus.begin_button_height);
  begin_btn->SetTexture(resources_->begin_button_texture);
  begin_btn->SetNormalSrcRect(
      {0, 0, ui_config_.menus.begin_button_texture_width,
       ui_config_.menus.begin_button_texture_height / 2});
  begin_btn->SetHoverSrcRect(
      {0, ui_config_.menus.begin_button_texture_height / 2,
       ui_config_.menus.begin_button_texture_width,
       ui_config_.menus.begin_button_texture_height / 2});

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
  black_bar->SetSize(kWindowWidth, static_cast<float>(kWindowHeight) / 3.0f);
  black_bar->SetBackgroundColor(WithOpacity(kColorBlack, 128));

  auto game_over_image = std::make_shared<UIImage>();
  game_over_image->SetId("game_over_image");
  game_over_image->SetAnchor(AnchorType::Center);
  game_over_image->SetSize(ui_config_.hud.game_over_sprite_width,
                           ui_config_.hud.game_over_sprite_height);
  game_over_image->SetTexture(resources_->game_over_texture);
  game_over_image->SetSrcRect({0, 0, ui_config_.hud.game_over_sprite_width,
                               ui_config_.hud.game_over_sprite_height});

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
  menu->SetSize(ui_config_.menus.quit_menu_width,
                ui_config_.menus.quit_menu_height);
  menu->SetBackground(resources_->settings_menu_background_texture);
  menu->SetBackgroundSrcRect(
      {0, 0, ui_config_.menus.settings_menu_background_sprite_width,
       ui_config_.menus.settings_menu_background_sprite_height});

  auto content = std::make_shared<VBox>();
  content->SetId("quit_confirm_content");
  content->SetSize(ui_config_.menus.quit_menu_width,
                   ui_config_.menus.quit_menu_height);
  content->SetPadding(ui_config_.menus.menu_content_padding);
  content->SetSpacing(ui_config_.menus.menu_item_spacing);

  // auto title_spacer = std::make_shared<Spacer>(0, 10);
  // content->AddChild(title_spacer);

  auto title = std::make_shared<UILabel>();
  title->SetId("quit_confirm_title");
  title->SetSize(ui_config_.menus.quit_menu_width -
                     2 * ui_config_.menus.menu_content_padding,
                 100);
  title->SetText("QUIT GAME?");
  title->SetFont(resources_->ui_font_huge);
  title->SetCenterWidth(ui_config_.menus.quit_menu_width -
                        2 * ui_config_.menus.menu_content_padding);
  content->AddChild(title);

  menu->AddChild(content);

  auto button_row = std::make_shared<HBox>();
  button_row->SetId("quit_confirm_buttons");
  button_row->SetAnchor(AnchorType::BottomCenter);
  button_row->SetPosition(0, -ui_config_.menus.menu_bottom_padding);
  button_row->SetSize(ui_config_.menus.settings_menu_button_width * 2 +
                          ui_config_.menus.menu_button_gap,
                      ui_config_.menus.settings_menu_button_height);
  button_row->SetSpacing(ui_config_.menus.menu_button_gap);

  auto yes_btn = std::make_shared<UIButton>();
  yes_btn->SetId("quit_yes_button");
  yes_btn->SetSize(ui_config_.menus.settings_menu_button_width,
                   ui_config_.menus.settings_menu_button_height);
  yes_btn->SetTexture(resources_->button_texture);
  yes_btn->SetNormalSrcRect(
      {0, 0, ui_config_.menus.generic_button_texture_width,
       ui_config_.menus.generic_button_texture_height / 2});
  yes_btn->SetHoverSrcRect(
      {0, ui_config_.menus.generic_button_texture_height / 2,
       ui_config_.menus.generic_button_texture_width,
       ui_config_.menus.generic_button_texture_height / 2});
  yes_btn->SetLabel("YES");
  yes_btn->SetLabelFont(resources_->ui_font_medium);
  button_row->AddChild(yes_btn);

  auto no_btn = std::make_shared<UIButton>();
  no_btn->SetId("quit_no_button");
  no_btn->SetSize(ui_config_.menus.settings_menu_button_width,
                  ui_config_.menus.settings_menu_button_height);
  no_btn->SetTexture(resources_->button_texture);
  no_btn->SetNormalSrcRect(
      {0, 0, ui_config_.menus.generic_button_texture_width,
       ui_config_.menus.generic_button_texture_height / 2});
  no_btn->SetHoverSrcRect({0,
                           ui_config_.menus.generic_button_texture_height / 2,
                           ui_config_.menus.generic_button_texture_width,
                           ui_config_.menus.generic_button_texture_height / 2});
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

// =============================================================================
// BuildChestOpeningScreen
// =============================================================================
void UIManager::BuildChestOpeningScreen() {
  auto overlay = std::make_shared<Panel>();
  overlay->SetId("chest_opening_menu");
  overlay->SetSize(kWindowWidth, kWindowHeight);
  overlay->SetBackgroundColor(WithOpacity(kColorBlack, 150));
  overlay->SetVisible(false);
  auto chest_animation = std::make_shared<UIAnimation>();
  chest_animation->SetId("chest_animated_image");
  chest_animation->SetAnchor(AnchorType::Center);
  chest_animation->SetSize(kChestAnimationWidth, kChestAnimationHeight);
  chest_animation->SetTexture(resources_->chest_texture);
  chest_animation->SetFrameDuration(kChestAnimationFrameDuration / 1000.0f);
  chest_animation->SetIsLoop(false);
  for (int frame_index = 0; frame_index < kChestTotalAnimFrames;
       ++frame_index) {
    int sprite_col, sprite_row;
    if (frame_index < kChestSpriteSheetCols) {
      sprite_col = frame_index;
      sprite_row = 0;
    } else {
      int looped_frame =
          (frame_index - kChestSpriteSheetCols) % (kChestSpriteSheetCols * 2);
      sprite_col = looped_frame % kChestSpriteSheetCols;
      sprite_row = 1 + (looped_frame / kChestSpriteSheetCols);
    }
    SDL_Rect src_rect = {sprite_col * kChestSpriteCellWidth,
                         sprite_row * kChestSpriteCellHeight,
                         kChestSpriteCellWidth, kChestSpriteCellHeight};
    chest_animation->AddFrame(src_rect);
  }
  overlay->AddChild(chest_animation);
  root_widget_->AddChild(overlay);
}

UIWidget* UIManager::GetChestOpeningRoot() {
  return root_widget_ ? root_widget_->FindWidget("chest_opening_menu")
                      : nullptr;
}

// =============================================================================
// BuildItemMenu: dynamically create card widgets from item options
// =============================================================================

void UIManager::BuildItemMenu(const UpgradeOptions& options) {
  root_widget_->RemoveChild("item_menu");

  auto overlay = std::make_shared<Panel>();
  overlay->SetId("item_menu");
  overlay->SetSize(kWindowWidth, kWindowHeight);
  overlay->SetBackgroundColor(WithOpacity(kColorBlack, 128));
  overlay->SetVisible(false);

  auto card_row = std::make_shared<HBox>();
  card_row->SetId("item_cards");
  card_row->SetAnchor(AnchorType::Center);
  int total_width = kNumItemOptions * ui_config_.cards.item_card_width +
                    (kNumItemOptions - 1) * ui_config_.cards.item_card_gap;
  card_row->SetSize(static_cast<float>(total_width),
                    ui_config_.cards.item_card_height);
  card_row->SetSpacing(ui_config_.cards.item_card_gap);

  for (size_t i = 0; i < options.size(); ++i) {
    BuildItemCard(card_row.get(), static_cast<int>(i),
                  static_cast<ItemUpgrade&>(*options[i]));
  }

  overlay->AddChild(card_row);
  root_widget_->AddChild(overlay);
  root_widget_->ComputeLayout(0, 0, kWindowWidth, kWindowHeight);
}

void UIManager::BuildItemCard(UIWidget* parent, int index,
                              const ItemUpgrade& upgrade) {
  std::string card_id = "item_card_" + std::to_string(index);

  auto card = std::make_shared<Panel>();
  card->SetId(card_id);
  card->SetSize(ui_config_.cards.item_card_width,
                ui_config_.cards.item_card_height);
  card->SetBackground(resources_->level_up_option_card_texture);
  card->SetBackgroundSrcRect({0, 0, 0, 0});  // full texture

  int item_id = static_cast<int>(upgrade.GetItemID());
  if (item_id >= 0 &&
      item_id < static_cast<int>(resources_->item_textures.size())) {
    auto icon = std::make_shared<UIImage>();
    icon->SetId(card_id + "_icon");
    icon->SetAnchor(AnchorType::TopCenter);
    icon->SetPosition(0, ui_config_.cards.item_card_icon_offset_y);
    icon->SetSize(ui_config_.cards.item_icon_size,
                  ui_config_.cards.item_icon_size);
    icon->SetTexture(resources_->item_textures[item_id]);
    icon->SetSrcRect({0, 0, 0, 0});
    card->AddChild(icon);
  }

  auto name_label = std::make_shared<UILabel>();
  name_label->SetId(card_id + "_name");
  name_label->SetPosition(ui_config_.cards.item_card_name_offset_x,
                          ui_config_.cards.item_card_name_offset_y);
  name_label->SetSize(ui_config_.cards.item_card_width -
                          2 * ui_config_.cards.item_card_name_offset_x,
                      96);
  name_label->SetText(upgrade.GetName());
  name_label->SetFont(resources_->ui_font_large);
  name_label->SetColor({255, 255, 255, 255});
  name_label->SetCenterWidth(ui_config_.cards.item_card_width -
                             2 * ui_config_.cards.item_card_name_offset_x);
  name_label->SetWrapWidth(ui_config_.cards.item_card_width -
                           2 * ui_config_.cards.item_card_name_offset_x);
  card->AddChild(name_label);

  std::vector<UpgradeDisplayRow> display_rows = upgrade.GetDisplayRows();
  for (size_t row_index = 0; row_index < display_rows.size(); ++row_index) {
    const UpgradeDisplayRow& row = display_rows[row_index];
    int row_desc_y =
        ui_config_.cards.item_card_desc_offset_y +
        static_cast<int>(row_index) * ui_config_.cards.item_card_row_stride;
    int row_stats_y =
        ui_config_.cards.item_card_stats_offset_y +
        static_cast<int>(row_index) * ui_config_.cards.item_card_row_stride;

    auto desc_label = std::make_shared<UILabel>();
    desc_label->SetId(card_id + "_desc_" + std::to_string(row_index));
    desc_label->SetPosition(ui_config_.cards.item_card_desc_offset_x,
                            static_cast<float>(row_desc_y));
    desc_label->SetSize(ui_config_.cards.item_card_width -
                            2 * ui_config_.cards.item_card_desc_offset_x,
                        25);
    desc_label->SetText(row.description);
    desc_label->SetFont(resources_->ui_font_medium);
    desc_label->SetColor({180, 180, 180, 255});
    desc_label->SetCenterWidth(ui_config_.cards.item_card_width -
                               2 * ui_config_.cards.item_card_desc_offset_x);
    card->AddChild(desc_label);

    if (row.old_value.empty() && row.new_value.empty()) {
      continue;
    }
    std::string stats_str = row.old_value + " -> " + row.new_value;
    SDL_Color stats_color = row.new_value > row.old_value
                                ? ui_config_.colors.positive_green
                                : ui_config_.colors.negative_red;
    auto stats_label = std::make_shared<UILabel>();
    stats_label->SetId(card_id + "_stats_" + std::to_string(row_index));
    stats_label->SetPosition(ui_config_.cards.item_card_stats_offset_x,
                             static_cast<float>(row_stats_y));
    stats_label->SetSize(ui_config_.cards.item_card_width -
                             2 * ui_config_.cards.item_card_stats_offset_x,
                         25);
    stats_label->SetText(stats_str);
    stats_label->SetFont(resources_->ui_font_medium);
    stats_label->SetColor(stats_color);
    stats_label->SetCenterWidth(ui_config_.cards.item_card_width -
                                2 * ui_config_.cards.item_card_stats_offset_x);
    card->AddChild(stats_label);
  }

  std::string btn_id = "select_button_" + std::to_string(index);
  auto select_btn = std::make_shared<UIButton>();
  select_btn->SetId(btn_id);
  select_btn->SetAnchor(AnchorType::TopCenter);
  select_btn->SetPosition(0, ui_config_.cards.item_card_button_offset_y);
  select_btn->SetSize(ui_config_.cards.item_card_button_width,
                      ui_config_.cards.item_card_button_height);
  select_btn->SetTexture(resources_->button_texture);
  select_btn->SetNormalSrcRect(
      {0, 0, ui_config_.menus.generic_button_texture_width,
       ui_config_.menus.generic_button_texture_height / 2});
  select_btn->SetHoverSrcRect(
      {0, ui_config_.menus.generic_button_texture_height / 2,
       ui_config_.menus.generic_button_texture_width,
       ui_config_.menus.generic_button_texture_height / 2});
  select_btn->SetLabel("CLAIM");
  select_btn->SetLabelFont(resources_->ui_font_medium);
  card->AddChild(select_btn);

  parent->AddChild(card);
}

void UIManager::UpdateItemMenu() {
  auto* item_menu = GetItemMenuRoot();
  if (!item_menu)
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
  update_hover(item_menu);
}

// =============================================================================
// BuildItemInventory: creates the inventory bar container
// =============================================================================

void UIManager::BuildItemInventory() {
  auto inventory_container = std::make_shared<Panel>();
  inventory_container->SetId("inventory_container");
  inventory_container->SetAnchor(AnchorType::TopCenter);
  inventory_container->SetPosition(
      0, static_cast<float>(ui_config_.inventory.inventory_bar_y));
  inventory_container->SetBackgroundColor(WithOpacity(
      kColorBlack, ui_config_.inventory.inventory_background_alpha));
  inventory_container->SetPadding(
      ui_config_.inventory.inventory_container_padding);
  inventory_container->SetVisible(false);

  auto inventory_bar = std::make_shared<HBox>();
  inventory_bar->SetId("inventory_bar");
  inventory_bar->SetSize(0, ui_config_.inventory.inventory_widget_height);
  inventory_bar->SetSpacing(ui_config_.inventory.inventory_item_gap);

  inventory_container->AddChild(inventory_bar);
  root_widget_->AddChild(inventory_container);
}

// =============================================================================
// BuildInventoryItem: creates a single inventory item widget
// ===========================================================================

void UIManager::BuildInventoryItem(UIWidget* parent, int index,
                                   const InventoryItem& inventory_item) {
  std::string item_id = "inventory_item_" + std::to_string(index);

  auto inventory_item_widget = std::make_shared<UIInventoryItem>();
  inventory_item_widget->SetId(item_id);
  int widget_width = ui_config_.inventory.inventory_icon_size +
                     ui_config_.inventory.inventory_label_width;
  inventory_item_widget->SetSize(static_cast<float>(widget_width),
                                 ui_config_.inventory.inventory_widget_height);
  inventory_item_widget->SetItemId(inventory_item.item_id);
  inventory_item_widget->SetItemCount(inventory_item.count);

  int item_texture_idx = static_cast<int>(inventory_item.item_id);
  if (item_texture_idx >= 0 &&
      item_texture_idx < static_cast<int>(resources_->item_textures.size())) {
    inventory_item_widget->SetItemTexture(
        resources_->item_textures[item_texture_idx]);
  }

  // Add multiplier label to the right of the icon
  auto multiplier_label = std::make_shared<UILabel>();
  multiplier_label->SetId(item_id + "_multiplier");
  std::string multiplier_text = "x" + std::to_string(inventory_item.count);
  multiplier_label->SetText(multiplier_text);
  multiplier_label->SetFont(resources_->ui_font_medium);
  multiplier_label->SetColor({255, 255, 255, 255});
  int label_x = ui_config_.inventory.inventory_icon_size +
                ui_config_.inventory.inventory_multiplier_margin;
  int label_y = (ui_config_.inventory.inventory_widget_height -
                 ui_config_.inventory.inventory_multiplier_size) /
                2;
  multiplier_label->SetPosition(static_cast<float>(label_x),
                                static_cast<float>(label_y));
  multiplier_label->SetSize(ui_config_.inventory.inventory_label_width,
                            ui_config_.inventory.inventory_multiplier_size);
  inventory_item_widget->AddChild(multiplier_label);

  parent->AddChild(inventory_item_widget);
}

// =============================================================================
// UpdateItemInventory: updates inventory bar with current items
// ===========================================================================

void UIManager::UpdateItemInventory(const Inventory& inventory) {
  auto* inventory_bar = root_widget_->FindWidgetAs<HBox>("inventory_bar");
  if (!inventory_bar) {
    return;
  }

  int index = 0;
  for (const auto& inventory_item : inventory) {
    std::string item_id = "inventory_item_" + std::to_string(index);
    auto inventory_item_widget =
        inventory_bar->FindWidgetAs<UIInventoryItem>(item_id);
    if (!inventory_item_widget) {
      BuildInventoryItem(inventory_bar, index, inventory_item);
    } else {
      inventory_item_widget->SetItemCount(inventory_item.count);
      auto label =
          inventory_item_widget->FindWidgetAs<UILabel>(item_id + "_multiplier");
      if (label) {
        label->SetText("x" + std::to_string(inventory_item.count));
      }
    }
    index++;
  }

  while (inventory_bar->GetChildren().size() > static_cast<size_t>(index)) {
    std::string child_id = inventory_bar->GetChildren().back()->GetId();
    inventory_bar->RemoveChild(child_id);
  }

  int num_items = static_cast<int>(inventory.size());
  int widget_width = ui_config_.inventory.inventory_icon_size +
                     ui_config_.inventory.inventory_label_width;
  int bar_width =
      num_items * widget_width +
      (num_items > 0 ? (num_items - 1) * ui_config_.inventory.inventory_item_gap
                     : 0);
  inventory_bar->SetSize(static_cast<float>(bar_width),
                         ui_config_.inventory.inventory_widget_height);

  int container_width =
      bar_width + 2 * ui_config_.inventory.inventory_container_padding;
  int container_height = ui_config_.inventory.inventory_widget_height +
                         2 * ui_config_.inventory.inventory_container_padding;
  auto* container = root_widget_->FindWidgetAs<Panel>("inventory_container");
  if (container) {
    container->SetSize(static_cast<float>(container_width),
                       static_cast<float>(container_height));
    container->SetVisible(num_items > 0);
  }

  root_widget_->ComputeLayout(0, 0, kWindowWidth, kWindowHeight);
}
}  // namespace arelto
