// include/action_manager.h
#ifndef RL2_ACTION_MANAGER_H_
#define RL2_ACTION_MANAGER_H_

#include "scene.h"

namespace rl2 {

class ActionManager {

 public:
  int GetActionSize(const Scene& scene);
  void ReadActionBuffer(const float* buffer_ptr, int buffer_size, Scene& scene);
};

}  // namespace rl2

#endif
