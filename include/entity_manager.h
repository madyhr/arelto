#ifndef RL2_ENTITY_MANAGER_H_
#define RL2_ENTITY_MANAGER_H_

#include <vector>
#include "event_manager.h"
#include "scene.h"

namespace arelto {

class EntityManager {
 public:
  EntityManager();
  ~EntityManager();

  void Initialize(Scene& scene, EventManager& event_manager);
  void Cleanup();
  void ProcessPendingSpawns();

 private:
  Scene* scene_ = nullptr;
  EventManager* event_manager_ = nullptr;
  std::vector<ExpGemData> pending_exp_gem_spawns_;
  std::vector<ChestData> pending_chest_spawns_;
  std::vector<int> pending_enemy_respawns_;

  void ResolveProjectileDestruction();
  void ResolveExpGemDestruction();
  void ResolveChestDestruction();

  void OnEnemyKilled(const EnemyKilledEvent& event, EventContext& context);
  void OnPlayerExpGemCollision(const PlayerExpGemCollisionEvent& event,
                               EventContext& context);
  void OnPlayerChestCollision(const PlayerChestCollisionEvent& event,
                              EventContext& context);
  void OnPlayerEnemyCollision(const PlayerEnemyCollisionEvent& event,
                              EventContext& context);
  void OnEnemyProjectileCollision(const EnemyProjectileCollisionEvent& event,
                                  EventContext& context);
};

}  // namespace arelto

#endif
