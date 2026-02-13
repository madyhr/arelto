// include/ui_manager.h
#ifndef RL2_UI_MANAGER_H_
#define RL2_UI_MANAGER_H_

#include <SDL2/SDL_render.h>
#include <SDL2/SDL_ttf.h>
#include <string>
#include <vector>
#include "scene.h"
#include "types.h"

namespace arelto {

struct UIResources {
  SDL_Texture* digit_font_texture = nullptr;
  SDL_Texture* health_bar_texture = nullptr;
  SDL_Texture* exp_bar_texture = nullptr;
  SDL_Texture* level_indicator_texture = nullptr;
  SDL_Texture* timer_hourglass_texture = nullptr;
  SDL_Texture* game_over_texture = nullptr;
  SDL_Texture* start_screen_texture = nullptr;
  SDL_Texture* paused_texture = nullptr;
  TTF_Font* ui_font_medium = nullptr;
  TTF_Font* ui_font_large = nullptr;
  TTF_Font* ui_font_huge = nullptr;
  SDL_Texture* level_up_option_card_texture = nullptr;
  SDL_Texture* button_texture = nullptr;
  SDL_Texture* begin_button_texture = nullptr;
  SDL_Texture* settings_menu_background_texture = nullptr;
  SDL_Texture* slider_texture = nullptr;
};

enum UIElementGroupType : int {
  health_bar = 0,
  exp_bar,
  level_indicator,
  timer,
  settings_menu,
};

struct UIElement {
  // position/size in texture atlas
  SDL_Rect src_rect;
  // offset relative to group
  Vector2D relative_offset;
  // size to actually draw on screen
  Size2D sprite_size;
  // tag for easier access
  enum Tag {
    none,
    background,
    fill,
    icon,
    text,
    button,
  } tag = none;

  // TODO: Refactor to a more polymorphic setup to not just use a "fat struct".
  Size2D char_size = {0, 0};
  std::string text_value = "";
};

struct UIElementGroup {
  UIElementGroupType type;
  Vector2D screen_position;
  std::vector<UIElement> elements;
  // TODO: Add is_visible property and a UIElementGroup by type func to uimanager,
  // to easily toggle the visibility of certain ui groups.
  // TODO: Find a better more generalizable way to define group tags useful for
  // UI modules.
  enum GroupTag {
    none,
    volume_control,
    main_menu_settings,
    resume_settings,
  } group_tag = none;

  UIElement* GetElemByTag(UIElement::Tag tag) {
    for (UIElement& el : elements) {
      if (el.tag == tag) {
        return &el;
      }
    }
    return nullptr;
  };
};

struct UIElementModule {
  UIElementGroupType type;
  Vector2D screen_position;
  std::vector<UIElement> module_elements;
  std::vector<UIElementGroup> element_groups;
  UIElementGroup* GetElemGroupByTag(UIElementGroup::GroupTag tag) {
    for (UIElementGroup& el_group : element_groups) {
      if (el_group.group_tag == tag) {
        return &el_group;
      }
    }
    return nullptr;
  };
};

class UIManager {

 public:
  UIElementGroup health_bar_;
  UIElementGroup exp_bar_;
  UIElementGroup timer_;
  UIElementGroup level_indicator_;
  UIElementModule settings_menu_;

  void SetupUI();
  void SetupHealthBar();
  void SetupExpBar();
  void SetupLevelIndicator();
  void SetupTimer();
  void SetupSettingsMenu();
  void UpdateUI(const Scene& scene, float time);
  void UpdateHealthBar(const Scene& scene);
  void UpdateExpBar(const Scene& scene);
  void UpdateLevelIndicator(const Scene& scene);
  void UpdateTimer(float time);
  void UpdateSettingsMenu(float volume, bool is_muted);

 private:
  void SetupSettingsBackground();
  void SetupSettingsVolumeControl();
  void SetupSettingsMainMenu();
  void SetupSettingsResume();
};

}  // namespace arelto

#endif
