// include/ui_manager.h
#ifndef RL2_UI_MANAGER_H_
#define RL2_UI_MANAGER_H_

#include <SDL2/SDL_render.h>
#include <SDL2/SDL_ttf.h>
#include <vector>
#include "scene.h"
#include "types.h"

namespace rl2 {

struct UIResources {
  SDL_Texture* digit_font_texture = nullptr;
  SDL_Texture* health_bar_texture = nullptr;
  SDL_Texture* timer_hourglass_texture = nullptr;
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
  Size sprite_size;
  // tag for easier access
  enum Tag { none, background, fill, icon, text } tag = none;
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
  // UIManager();
  // ~UIManager();
  UIElementGroup health_bar_;
  UIElementGroup timer_;

  void SetupUI();
  void SetupHealthBar();
  void SetupTimer();
  void UpdateUI(const Scene& scene);
  void UpdateHealthBar(int current_hp, int max_hp);
  void UpdateTimer(float time);

 private:
};

}  // namespace rl2

#endif
