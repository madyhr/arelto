// include/game.h
#ifndef RL2_UTILS_H_
#define RL2_UTILS_H_

#include <array>
#include "constants/game.h"

namespace arelto {

class FrameStats {
 public:
  std::array<float, kFrameTimes> frame_time_buffer{};
  float frame_time_sum = 0.0f;
  int current_buffer_length = 0;
  int max_buffer_length = 100;
  int head_index = 0;
  void update_frame_time_buffer(float dt);
  float get_average_frame_time();
  void print_fps_running_average(float dt);
};

}  // namespace arelto

#endif
