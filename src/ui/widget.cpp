// src/ui/widget.cpp
#include "ui/widget.h"
#include <algorithm>

namespace arelto {

void UIWidget::AddChild(std::shared_ptr<UIWidget> child) {
  child->parent_ = this;
  children_.push_back(std::move(child));
}

void UIWidget::RemoveChild(const std::string& id) {
  children_.erase(std::remove_if(children_.begin(), children_.end(),
                                 [&id](const std::shared_ptr<UIWidget>& child) {
                                   return child->GetId() == id;
                                 }),
                  children_.end());
}

UIWidget* UIWidget::GetParent() const {
  return parent_;
}

const std::vector<std::shared_ptr<UIWidget>>& UIWidget::GetChildren() const {
  return children_;
}

void UIWidget::SetPosition(float x, float y) {
  pos_x_ = x;
  pos_y_ = y;
}

void UIWidget::SetSize(float w, float h) {
  width_ = w;
  height_ = h;
}

void UIWidget::SetAnchor(AnchorType anchor) {
  anchor_ = anchor;
}
void UIWidget::SetPadding(float padding) {
  padding_ = padding;
}
void UIWidget::SetMargin(float margin) {
  margin_ = margin;
}
void UIWidget::SetSpacing(float spacing) {
  spacing_ = spacing;
}

void UIWidget::SetId(const std::string& id) {
  id_ = id;
}
const std::string& UIWidget::GetId() const {
  return id_;
}

void UIWidget::SetVisible(bool visible) {
  visible_ = visible;
}
bool UIWidget::IsVisible() const {
  return visible_;
}

void UIWidget::SetHovered(bool hovered) {
  hovered_ = hovered;
}
bool UIWidget::IsHovered() const {
  return hovered_;
}

SDL_Rect UIWidget::GetComputedBounds() const {
  return computed_bounds_;
}

WidgetType UIWidget::GetWidgetType() const {
  return WidgetType::Base;
}

void UIWidget::ApplyAnchor(int parent_x, int parent_y, int parent_w,
                           int parent_h) {
  int w = static_cast<int>(width_);
  int h = static_cast<int>(height_);

  int base_x = parent_x;
  int base_y = parent_y;

  switch (anchor_) {
    case AnchorType::TopLeft:
      break;
    case AnchorType::TopCenter:
      base_x = parent_x + (parent_w - w) / 2;
      break;
    case AnchorType::TopRight:
      base_x = parent_x + parent_w - w;
      break;
    case AnchorType::CenterLeft:
      base_y = parent_y + (parent_h - h) / 2;
      break;
    case AnchorType::Center:
      base_x = parent_x + (parent_w - w) / 2;
      base_y = parent_y + (parent_h - h) / 2;
      break;
    case AnchorType::CenterRight:
      base_x = parent_x + parent_w - w;
      base_y = parent_y + (parent_h - h) / 2;
      break;
    case AnchorType::BottomLeft:
      base_y = parent_y + parent_h - h;
      break;
    case AnchorType::BottomCenter:
      base_x = parent_x + (parent_w - w) / 2;
      base_y = parent_y + parent_h - h;
      break;
    case AnchorType::BottomRight:
      base_x = parent_x + parent_w - w;
      base_y = parent_y + parent_h - h;
      break;
  }

  computed_bounds_.x =
      base_x + static_cast<int>(pos_x_) + static_cast<int>(margin_);
  computed_bounds_.y =
      base_y + static_cast<int>(pos_y_) + static_cast<int>(margin_);
  computed_bounds_.w = w;
  computed_bounds_.h = h;
}

void UIWidget::ComputeLayout(int parent_x, int parent_y, int parent_w,
                             int parent_h) {
  ApplyAnchor(parent_x, parent_y, parent_w, parent_h);

  int content_x = computed_bounds_.x + static_cast<int>(padding_);
  int content_y = computed_bounds_.y + static_cast<int>(padding_);
  int content_w = computed_bounds_.w - 2 * static_cast<int>(padding_);
  int content_h = computed_bounds_.h - 2 * static_cast<int>(padding_);

  for (auto& child : children_) {
    child->ComputeLayout(content_x, content_y, content_w, content_h);
  }
}

void UIWidget::Update(float dt) {
  for (auto& child : children_) {
    if (child->IsVisible()) {
      child->Update(dt);
    }
  }
}

UIWidget* UIWidget::FindWidget(const std::string& id) {
  if (id_ == id) {
    return this;
  }
  for (auto& child : children_) {
    UIWidget* found = child->FindWidget(id);
    if (found) {
      return found;
    }
  }
  return nullptr;
}

}  // namespace arelto
