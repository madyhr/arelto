// include/ui/containers.h
#ifndef RL2_UI_CONTAINERS_H_
#define RL2_UI_CONTAINERS_H_

#include <SDL2/SDL_render.h>
#include "ui/widget.h"

namespace arelto {

// Panel: a container with an optional background texture.
class Panel : public UIWidget {
 public:
  Panel() = default;

  void SetBackground(SDL_Texture* texture);
  void SetBackgroundSrcRect(SDL_Rect src_rect);

  SDL_Texture* GetBackgroundTexture() const;
  SDL_Rect GetBackgroundSrcRect() const;

  WidgetType GetWidgetType() const override;

  void SetBackgroundColor(SDL_Color color);
  SDL_Color GetBackgroundColor() const;
  bool HasBackgroundColor() const;

 private:
  SDL_Texture* background_texture_ = nullptr;
  SDL_Rect background_src_rect_ = {0, 0, 0, 0};
  bool has_custom_src_rect_ = false;

  SDL_Color background_color_ = {0, 0, 0, 0};
  bool use_background_color_ = false;
};

// VBox: stacks children vertically with spacing.
class VBox : public UIWidget {
 public:
  VBox() = default;

  void ComputeLayout(int parent_x, int parent_y, int parent_w,
                     int parent_h) override;
  WidgetType GetWidgetType() const override;
};

// HBox: arranges children horizontally with spacing.
class HBox : public UIWidget {
 public:
  HBox() = default;

  void ComputeLayout(int parent_x, int parent_y, int parent_w,
                     int parent_h) override;
  WidgetType GetWidgetType() const override;
};

}  // namespace arelto

#endif
