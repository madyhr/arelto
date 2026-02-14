// src/ui/widgets.cpp
#include "ui/widgets.h"
#include <algorithm>

namespace arelto {

// =============================================================================
// UIImage
// =============================================================================

void UIImage::SetTexture(SDL_Texture* texture) { texture_ = texture; }
void UIImage::SetSrcRect(SDL_Rect src_rect) { src_rect_ = src_rect; }
SDL_Texture* UIImage::GetTexture() const { return texture_; }
SDL_Rect UIImage::GetSrcRect() const { return src_rect_; }
WidgetType UIImage::GetWidgetType() const { return WidgetType::Image; }

// =============================================================================
// UILabel
// =============================================================================

void UILabel::SetText(const std::string& text) { text_ = text; }
const std::string& UILabel::GetText() const { return text_; }
void UILabel::SetFont(TTF_Font* font) { font_ = font; }
TTF_Font* UILabel::GetFont() const { return font_; }
void UILabel::SetColor(SDL_Color color) { color_ = color; }
SDL_Color UILabel::GetColor() const { return color_; }
void UILabel::SetUseDigitFont(bool use_digit_font) {
  use_digit_font_ = use_digit_font;
}
bool UILabel::GetUseDigitFont() const { return use_digit_font_; }

void UILabel::SetCharSize(int width, int height) {
  char_width_ = width;
  char_height_ = height;
}
int UILabel::GetCharWidth() const { return char_width_; }
int UILabel::GetCharHeight() const { return char_height_; }

void UILabel::SetDigitSpriteSize(int width, int height) {
  digit_sprite_width_ = width;
  digit_sprite_height_ = height;
}
int UILabel::GetDigitSpriteWidth() const { return digit_sprite_width_; }
int UILabel::GetDigitSpriteHeight() const { return digit_sprite_height_; }

void UILabel::SetCenterWidth(int width) { center_width_ = width; }
int UILabel::GetCenterWidth() const { return center_width_; }

WidgetType UILabel::GetWidgetType() const { return WidgetType::Label; }

// =============================================================================
// UIButton
// =============================================================================

void UIButton::SetTexture(SDL_Texture* texture) { texture_ = texture; }
SDL_Texture* UIButton::GetTexture() const { return texture_; }

void UIButton::SetNormalSrcRect(SDL_Rect src_rect) {
  normal_src_rect_ = src_rect;
}
SDL_Rect UIButton::GetNormalSrcRect() const { return normal_src_rect_; }

void UIButton::SetHoverSrcRect(SDL_Rect src_rect) {
  hover_src_rect_ = src_rect;
}
SDL_Rect UIButton::GetHoverSrcRect() const { return hover_src_rect_; }

SDL_Rect UIButton::GetCurrentSrcRect() const {
  return hovered_ ? hover_src_rect_ : normal_src_rect_;
}

void UIButton::SetLabel(const std::string& label) { label_ = label; }
const std::string& UIButton::GetLabel() const { return label_; }

void UIButton::SetLabelFont(TTF_Font* font) { label_font_ = font; }
TTF_Font* UIButton::GetLabelFont() const { return label_font_; }

void UIButton::SetOnClick(std::function<void()> callback) {
  on_click_ = std::move(callback);
}
const std::function<void()>& UIButton::GetOnClick() const { return on_click_; }

WidgetType UIButton::GetWidgetType() const { return WidgetType::Button; }

// =============================================================================
// UIProgressBar
// =============================================================================

void UIProgressBar::SetPercent(float percent) {
  percent_ = std::clamp(percent, 0.0f, 1.0f);
}
float UIProgressBar::GetPercent() const { return percent_; }

void UIProgressBar::SetContainerTexture(SDL_Texture* texture) {
  container_texture_ = texture;
}
SDL_Texture* UIProgressBar::GetContainerTexture() const {
  return container_texture_;
}
void UIProgressBar::SetContainerSrcRect(SDL_Rect src_rect) {
  container_src_rect_ = src_rect;
}
SDL_Rect UIProgressBar::GetContainerSrcRect() const {
  return container_src_rect_;
}

void UIProgressBar::SetFillTexture(SDL_Texture* texture) {
  fill_texture_ = texture;
}
SDL_Texture* UIProgressBar::GetFillTexture() const { return fill_texture_; }
void UIProgressBar::SetFillSrcRect(SDL_Rect src_rect) {
  fill_src_rect_ = src_rect;
}
SDL_Rect UIProgressBar::GetFillSrcRect() const { return fill_src_rect_; }

void UIProgressBar::SetFillOffset(int x, int y) {
  fill_offset_x_ = x;
  fill_offset_y_ = y;
}
int UIProgressBar::GetFillOffsetX() const { return fill_offset_x_; }
int UIProgressBar::GetFillOffsetY() const { return fill_offset_y_; }

void UIProgressBar::SetMaxFillSize(int width, int height) {
  max_fill_width_ = width;
  max_fill_height_ = height;
}
int UIProgressBar::GetMaxFillWidth() const { return max_fill_width_; }
int UIProgressBar::GetMaxFillHeight() const { return max_fill_height_; }

SDL_Rect UIProgressBar::GetClippedFillSrcRect() const {
  SDL_Rect clipped = fill_src_rect_;
  clipped.w = static_cast<int>(fill_src_rect_.w * percent_);
  return clipped;
}

SDL_Rect UIProgressBar::GetFillDestRect() const {
  SDL_Rect dest;
  dest.x = computed_bounds_.x + fill_offset_x_;
  dest.y = computed_bounds_.y + fill_offset_y_;
  dest.w = static_cast<int>(max_fill_width_ * percent_);
  dest.h = max_fill_height_;
  return dest;
}

WidgetType UIProgressBar::GetWidgetType() const {
  return WidgetType::ProgressBar;
}

}  // namespace arelto
