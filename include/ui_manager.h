// include/ui_manager.h
#ifndef RL2_UI_MANAGER_H_
#define RL2_UI_MANAGER_H_

#include <SDL2/SDL_render.h>
#include <SDL2/SDL_ttf.h>
#include <memory>
#include <string>
#include <vector>
#include "items.h"
#include "scene.h"
#include "ui/widget.h"
#include "upgrades.h"

namespace arelto {

struct UIResources {
  SDL_Texture* digit_font_texture = nullptr;
  SDL_Texture* health_bar_texture = nullptr;
  SDL_Texture* exp_bar_texture = nullptr;
  SDL_Texture* level_indicator_texture = nullptr;
  SDL_Texture* timer_hourglass_texture = nullptr;
  SDL_Texture* game_over_texture = nullptr;
  SDL_Texture* start_screen_texture = nullptr;
  TTF_Font* ui_font_medium = nullptr;
  TTF_Font* ui_font_large = nullptr;
  TTF_Font* ui_font_huge = nullptr;
  SDL_Texture* level_up_option_card_texture = nullptr;
  SDL_Texture* button_texture = nullptr;
  SDL_Texture* begin_button_texture = nullptr;
  SDL_Texture* settings_menu_background_texture = nullptr;
  SDL_Texture* slider_texture = nullptr;
  SDL_Texture* checkbox_texture = nullptr;
  SDL_Texture* checkmark_texture = nullptr;
  SDL_Texture* chest_texture = nullptr;
  std::vector<SDL_Texture*> projectile_textures;
  std::vector<SDL_Texture*> item_textures;
};

class UIManager {
 public:
  void SetupUI(const UIResources& resources);
  void Update(const Scene& scene, float time);
  void UpdateSettingsMenu(float volume, bool is_muted,
                          const GameStatus& game_status);
  void BuildLevelUpMenu(const UpgradeOptions& options);
  void UpdateLevelUpMenu();
  void BuildStartScreen();
  void UpdateStartScreen();
  void BuildGameOverScreen();
  void UpdateQuitConfirmMenu();
  void BuildChestOpeningScreen();
  void BuildItemMenu(const UpgradeOptions& options);
  void UpdateItemMenu();

  UIWidget* GetRootWidget();
  UIWidget* GetSettingsRoot();
  UIWidget* GetLevelUpRoot();
  UIWidget* GetItemMenuRoot();
  UIWidget* GetStartScreenRoot();
  UIWidget* GetGameOverScreenRoot();
  UIWidget* GetQuitConfirmRoot();
  UIWidget* GetChestOpeningRoot();

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
  void BuildQuitConfirmMenu();
  void BuildLevelUpCard(UIWidget* parent, int index,
                        const SpellStatUpgrade& upgrade);
  void BuildItemCard(UIWidget* parent, int index, const ItemUpgrade& upgrade);
};

}  // namespace arelto

#endif
