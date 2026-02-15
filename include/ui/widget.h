// include/ui/widget.h
#ifndef RL2_UI_WIDGET_H_
#define RL2_UI_WIDGET_H_

#include <SDL2/SDL_rect.h>
#include <memory>
#include <string>
#include <vector>

namespace arelto {

enum class AnchorType {
  TopLeft,
  TopCenter,
  TopRight,
  CenterLeft,
  Center,
  CenterRight,
  BottomLeft,
  BottomCenter,
  BottomRight,
};

enum class WidgetType {
  Base,
  Panel,
  VBox,
  HBox,
  Image,
  Label,
  Button,
  ProgressBar,
  Checkbox,
  Spacer,
};

class UIWidget {
 public:
  virtual ~UIWidget() = default;

  void AddChild(std::shared_ptr<UIWidget> child);
  void RemoveChild(const std::string& id);
  UIWidget* GetParent() const;
  const std::vector<std::shared_ptr<UIWidget>>& GetChildren() const;

  void SetPosition(float x, float y);
  void SetSize(float w, float h);
  void SetAnchor(AnchorType anchor);
  void SetPadding(float padding);
  void SetMargin(float margin);
  void SetSpacing(float spacing);

  void SetId(const std::string& id);
  const std::string& GetId() const;

  void SetVisible(bool visible);
  bool IsVisible() const;
  void SetHovered(bool hovered);
  bool IsHovered() const;

  SDL_Rect GetComputedBounds() const;

  // Layout computation — resolves positions based on parent/children
  virtual void ComputeLayout(int parent_x, int parent_y, int parent_w,
                             int parent_h);

  // Update — called each frame to propagate state changes
  virtual void Update(float dt);

  virtual WidgetType GetWidgetType() const;

  // Widget lookup by ID (recursive)
  UIWidget* FindWidget(const std::string& id);

  template <typename T>
  T* FindWidgetAs(const std::string& id) {
    return dynamic_cast<T*>(FindWidget(id));
  }

 protected:
  std::string id_;
  SDL_Rect computed_bounds_ = {0, 0, 0, 0};
  std::vector<std::shared_ptr<UIWidget>> children_;
  UIWidget* parent_ = nullptr;

  float pos_x_ = 0.0f;
  float pos_y_ = 0.0f;
  float width_ = 0.0f;
  float height_ = 0.0f;
  AnchorType anchor_ = AnchorType::TopLeft;
  float padding_ = 0.0f;
  float margin_ = 0.0f;
  float spacing_ = 0.0f;

  bool visible_ = true;
  bool hovered_ = false;

  // Compute anchored position within parent using `anchor_` attribute
  void ApplyAnchor(int parent_x, int parent_y, int parent_w, int parent_h);
};

}  // namespace arelto

#endif
