// include/ui_manager.h
#ifndef RL2_UI_MANAGER_H_
#define RL2_UI_MANAGER_H_

#include <SDL2/SDL_render.h>
#include <SDL2/SDL_ttf.h>
#include <memory>
#include <string>
#include <vector>
#include "config/ui_config.h"
#include "event_manager.h"
#include "items.h"
#include "render_resources.h"
#include "ui/widget.h"
#include "upgrades.h"

namespace arelto {

class UIManager {
 public:
  void SetupUI(const RenderResources& resources, const UIConfig& config,
               EventManager& event_manager);
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
  void BuildItemInventory();
  void UpdateTimer(float time);

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
  const RenderResources* resources_ = nullptr;
  UIConfig ui_config_ = MakeDefaultUIConfig();

  void BuildHUD();
  void UpdateExpBar(int current_exp_points, int exp_points_required);
  void UpdateHealthBar(int health_points, int max_health_points);
  void UpdateItemInventory(const Inventory& inventory);
  void SetupUIEventSubscriptions(EventManager& event_manager);
  void BuildSettingsMenu();
  void BuildQuitConfirmMenu();
  void BuildLevelUpCard(UIWidget* parent, int index,
                        const SpellStatUpgrade& upgrade);
  void BuildItemCard(UIWidget* parent, int index, const ItemUpgrade& upgrade);
  void BuildInventoryItem(UIWidget* parent, int index,
                          const InventoryItem& inv_item);
};

}  // namespace arelto

#endif
