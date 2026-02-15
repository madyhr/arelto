// src/ui/containers.cpp
#include "ui/containers.h"

namespace arelto {

// =============================================================================
// Panel
// =============================================================================

void Panel::SetBackground(SDL_Texture* texture) {
  background_texture_ = texture;
}

void Panel::SetBackgroundSrcRect(SDL_Rect src_rect) {
  background_src_rect_ = src_rect;
  has_custom_src_rect_ = true;
}

SDL_Texture* Panel::GetBackgroundTexture() const {
  return background_texture_;
}

SDL_Rect Panel::GetBackgroundSrcRect() const {
  if (has_custom_src_rect_) {
    return background_src_rect_;
  }
  // Default: use the entire texture
  return {0, 0, computed_bounds_.w, computed_bounds_.h};
}

void Panel::SetBackgroundColor(SDL_Color color) {
  background_color_ = color;
  use_background_color_ = true;
}

SDL_Color Panel::GetBackgroundColor() const {
  return background_color_;
}

bool Panel::HasBackgroundColor() const {
  return use_background_color_;
}

WidgetType Panel::GetWidgetType() const {
  return WidgetType::Panel;
}

// =============================================================================
// VBox
// =============================================================================

void VBox::ComputeLayout(int parent_x, int parent_y, int parent_w,
                         int parent_h) {
  ApplyAnchor(parent_x, parent_y, parent_w, parent_h);

  int content_x = computed_bounds_.x + static_cast<int>(padding_);
  int content_y = computed_bounds_.y + static_cast<int>(padding_);
  int content_w = computed_bounds_.w - 2 * static_cast<int>(padding_);
  int content_h = computed_bounds_.h - 2 * static_cast<int>(padding_);

  int current_y = content_y;

  for (auto& child : children_) {
    child->ComputeLayout(content_x, current_y, content_w, content_h);
    current_y += child->GetComputedBounds().h + static_cast<int>(spacing_);
  }
}

WidgetType VBox::GetWidgetType() const {
  return WidgetType::VBox;
}

// =============================================================================
// HBox
// =============================================================================

void HBox::ComputeLayout(int parent_x, int parent_y, int parent_w,
                         int parent_h) {
  ApplyAnchor(parent_x, parent_y, parent_w, parent_h);

  int content_x = computed_bounds_.x + static_cast<int>(padding_);
  int content_y = computed_bounds_.y + static_cast<int>(padding_);
  int content_w = computed_bounds_.w - 2 * static_cast<int>(padding_);
  int content_h = computed_bounds_.h - 2 * static_cast<int>(padding_);

  int current_x = content_x;

  for (auto& child : children_) {
    child->ComputeLayout(current_x, content_y, content_w, content_h);
    current_x += child->GetComputedBounds().w + static_cast<int>(spacing_);
  }
}

WidgetType HBox::GetWidgetType() const {
  return WidgetType::HBox;
}

}  // namespace arelto
