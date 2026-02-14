// include/ui_manager.h
#ifndef RL2_UI_MANAGER_H_
#define RL2_UI_MANAGER_H_

#include <SDL2/SDL_render.h>
#include <SDL2/SDL_ttf.h>
#include <memory>
#include <string>
#include <vector>
#include "scene.h"
#include "ui/containers.h"
#include "ui/widget.h"
#include "ui/widgets.h"

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
  std::vector<SDL_Texture*> projectile_textures;
};

class UIManager {
 public:
  void SetupUI(const UIResources& resources);
  void Update(const Scene& scene, float time);
  void UpdateSettingsMenu(float volume, bool is_muted);
  void BuildLevelUpMenu(const std::vector<std::unique_ptr<Upgrade>>& options);
  void UpdateLevelUpMenu();
  void BuildStartScreen();
  void UpdateStartScreen();
  void BuildGameOverScreen();

  UIWidget* GetRootWidget();
  UIWidget* GetSettingsRoot();
  UIWidget* GetLevelUpRoot();
  UIWidget* GetStartScreenRoot();
  UIWidget* GetGameOverScreenRoot();

  template <typename T>
  T* GetWidget(const std::string& id) {
    if (!root_widget_) {
      return nullptr;
    }
    return root_widget_->FindWidgetAs<T>(id);
  }

 private:
  std::shared_ptr<UIWidget> root_widget_;
  const UIResources* resources_ = nullptr;

  void BuildHUD();
  void BuildSettingsMenu();
  void BuildLevelUpCard(UIWidget* parent, int index, const Upgrade& upgrade);
};

}  // namespace arelto

#endif
