// include/observation_manager.h
#ifndef RL2_OBSERVATION_MANAGER_H_
#define RL2_OBSERVATION_MANAGER_H_

#include "scene.h"

namespace arelto {

class ObservationManager {

 public:
  int GetObservationSize(const Scene& scene);
  void FillObservationBuffer(float* buffer_ptr, int buffer_size,
                             const Scene& scene);
};

}  // namespace arelto

#endif
