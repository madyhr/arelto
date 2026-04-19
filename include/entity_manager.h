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

  void Initialize(EventManager& event_manager);
  void Cleanup(Scene& scene);
  void ProcessPendingSpawns(Scene& scene);

 private:
  EventManager* event_manager_ = nullptr;
  std::vector<ExpGemData> pending_exp_gem_spawns_;
  std::vector<ChestData> pending_chest_spawns_;
  std::vector<int> pending_enemy_respawns_;

  void ResolveProjectileDestruction(Scene& scene);
  void ResolveExpGemDestruction(Scene& scene);
  void ResolveChestDestruction(Scene& scene);

  void OnEnemyKilled(const EnemyKilledEvent& event, EventContext& context);
  void OnPlayerExpGemCollision(const PlayerExpGemCollisionEvent& event,
                               EventContext& context);
  void OnPlayerChestCollision(const PlayerChestCollisionEvent& event,
                              EventContext& context);
  void OnPlayerDamaged(const PlayerDamagedEvent& event, EventContext& context);
  void OnPlayerEnemyCollision(const PlayerEnemyCollisionEvent& event,
                              EventContext& context);
  void OnEnemyDamaged(const EnemyDamagedEvent& event, EventContext& context);
  void OnEnemyProjectileCollision(const EnemyProjectileCollisionEvent& event,
                                  EventContext& context);
};

}  // namespace arelto

#endif
