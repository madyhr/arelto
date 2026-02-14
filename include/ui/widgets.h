// include/ui/widgets.h
#ifndef RL2_UI_WIDGETS_H_
#define RL2_UI_WIDGETS_H_

#include <SDL2/SDL_render.h>
#include <SDL2/SDL_ttf.h>
#include <functional>
#include <string>
#include "ui/widget.h"

namespace arelto {

// UIImage: renders a texture or a region of a texture atlas.
class UIImage : public UIWidget {
 public:
  UIImage() = default;

  void SetTexture(SDL_Texture* texture);
  void SetSrcRect(SDL_Rect src_rect);

  SDL_Texture* GetTexture() const;
  SDL_Rect GetSrcRect() const;

  WidgetType GetWidgetType() const override;

 private:
  SDL_Texture* texture_ = nullptr;
  SDL_Rect src_rect_ = {0, 0, 0, 0};
};

// UILabel: renders text using either TTF fonts or a digit sprite sheet.
class UILabel : public UIWidget {
 public:
  UILabel() = default;

  void SetText(const std::string& text);
  const std::string& GetText() const;

  void SetFont(TTF_Font* font);
  TTF_Font* GetFont() const;

  void SetColor(SDL_Color color);
  SDL_Color GetColor() const;

  // When true, uses the digit sprite sheet instead of TTF.
  void SetUseDigitFont(bool use_digit_font);
  bool GetUseDigitFont() const;

  void SetCharSize(int width, int height);
  int GetCharWidth() const;
  int GetCharHeight() const;

  // Sprite size for digit font (size of each cell in the sprite sheet)
  void SetDigitSpriteSize(int width, int height);
  int GetDigitSpriteWidth() const;
  int GetDigitSpriteHeight() const;

  // If > 0, text is centered within this width.
  void SetCenterWidth(int width);
  int GetCenterWidth() const;

  WidgetType GetWidgetType() const override;

 private:
  std::string text_;
  TTF_Font* font_ = nullptr;
  SDL_Color color_ = {255, 255, 255, 255};
  bool use_digit_font_ = false;
  int char_width_ = 0;
  int char_height_ = 0;
  int digit_sprite_width_ = 0;
  int digit_sprite_height_ = 0;
  int center_width_ = 0;
};

// UIButton: a texture with normal/hover states and an optional label + click
// callback.
class UIButton : public UIWidget {
 public:
  UIButton() = default;

  void SetTexture(SDL_Texture* texture);
  SDL_Texture* GetTexture() const;

  // Source rect for normal state (top half of texture by convention)
  void SetNormalSrcRect(SDL_Rect src_rect);
  SDL_Rect GetNormalSrcRect() const;

  // Source rect for hovered state (bottom half of texture by convention)
  void SetHoverSrcRect(SDL_Rect src_rect);
  SDL_Rect GetHoverSrcRect() const;

  // Returns the appropriate src rect based on hover state
  SDL_Rect GetCurrentSrcRect() const;

  void SetLabel(const std::string& label);
  const std::string& GetLabel() const;

  void SetLabelFont(TTF_Font* font);
  TTF_Font* GetLabelFont() const;

  void SetOnClick(std::function<void()> callback);
  const std::function<void()>& GetOnClick() const;

  WidgetType GetWidgetType() const override;

 private:
  SDL_Texture* texture_ = nullptr;
  SDL_Rect normal_src_rect_ = {0, 0, 0, 0};
  SDL_Rect hover_src_rect_ = {0, 0, 0, 0};
  std::string label_;
  TTF_Font* label_font_ = nullptr;
  std::function<void()> on_click_;
};

// UIProgressBar: handles container + fill textures with clipping based on a
// percentage value.
class UIProgressBar : public UIWidget {
 public:
  UIProgressBar() = default;

  void SetPercent(float percent);
  float GetPercent() const;

  // Container (background) texture
  void SetContainerTexture(SDL_Texture* texture);
  SDL_Texture* GetContainerTexture() const;
  void SetContainerSrcRect(SDL_Rect src_rect);
  SDL_Rect GetContainerSrcRect() const;

  // Fill texture
  void SetFillTexture(SDL_Texture* texture);
  SDL_Texture* GetFillTexture() const;
  void SetFillSrcRect(SDL_Rect src_rect);
  SDL_Rect GetFillSrcRect() const;

  // Fill offset relative to container
  void SetFillOffset(int x, int y);
  int GetFillOffsetX() const;
  int GetFillOffsetY() const;

  // Max fill dimensions
  void SetMaxFillSize(int width, int height);
  int GetMaxFillWidth() const;
  int GetMaxFillHeight() const;

  // Returns the clipped fill src rect based on current percent
  SDL_Rect GetClippedFillSrcRect() const;
  // Returns the fill destination rect (position + clipped width)
  SDL_Rect GetFillDestRect() const;

  WidgetType GetWidgetType() const override;

 private:
  float percent_ = 1.0f;
  SDL_Texture* container_texture_ = nullptr;
  SDL_Rect container_src_rect_ = {0, 0, 0, 0};
  SDL_Texture* fill_texture_ = nullptr;
  SDL_Rect fill_src_rect_ = {0, 0, 0, 0};
  int fill_offset_x_ = 0;
  int fill_offset_y_ = 0;
  int max_fill_width_ = 0;
  int max_fill_height_ = 0;
};

}  // namespace arelto

#endif
