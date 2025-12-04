// include/ui_manager.h
#ifndef RL2_UI_MANAGER_H_
#define RL2_UI_MANAGER_H_

#include <SDL2/SDL_render.h>
#include <SDL2/SDL_ttf.h>
#include <string>
#include <vector>
#include "scene.h"
#include "types.h"

namespace rl2 {

struct UIResources {
  SDL_Texture* digit_font_texture = nullptr;
  SDL_Texture* health_bar_texture = nullptr;
  SDL_Texture* timer_hourglass_texture = nullptr;
  SDL_Texture* game_over_texture = nullptr;
};

enum UIElementGroupType : int {
  health_bar = 0,
  timer,
};

struct UIElement {
  // position/size in texture atlas
  SDL_Rect src_rect;
  // offset relative to group
  Vector2D relative_offset;
  // size to actually draw on screen
  Size2D sprite_size;
  // tag for easier access
  enum Tag { none, background, fill, icon, text } tag = none;

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

  UIElement* GetElemByTag(UIElement::Tag tag) {
    for (UIElement& el : elements) {
      if (el.tag == tag) {
        return &el;
      }
    }
    return nullptr;
  };
};

class UIManager {

 public:
  UIElementGroup health_bar_;
  UIElementGroup timer_;

  void SetupUI();
  void SetupHealthBar();
  void SetupTimer();
  void UpdateUI(const Scene& scene, float time);
  void UpdateHealthBar(const Scene& scene);
  void UpdateTimer(float time);

 private:
};

}  // namespace rl2

#endif
