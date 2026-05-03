// src/utils.cpp
#include "utils.h"
#include <iostream>

namespace arelto {

void FrameStats::update_frame_time_buffer(float new_value) {
  float oldest_value = frame_time_buffer[head_index];
  frame_time_sum = frame_time_sum - oldest_value + new_value;
  frame_time_buffer[head_index] = new_value;

  if (current_buffer_length < max_buffer_length) {
    current_buffer_length++;
  };

  head_index = (head_index + 1) % max_buffer_length;
};

float FrameStats::get_average_frame_time() {
  if (current_buffer_length == 0) {
    return 0.0f;
  }
  return frame_time_sum / static_cast<float>(current_buffer_length);
};

void FrameStats::print_fps_running_average(float dt) {
  static float accumulated_time = 0.0f;
  update_frame_time_buffer(dt);
  if (accumulated_time > 1.0f) {
    float avg_frame_time = get_average_frame_time();
    float average_fps = 1.0f / (avg_frame_time);
    std::cout << "Current Avg FPS: " << std::fixed << average_fps;
    std::cout << "\r" << std::flush;
    accumulated_time -= 1.0f;
  };
  accumulated_time += dt;
};

std::string ToTitleCase(std::string text) {
  bool capitalize_next = true;

  for (char& c : text) {
    unsigned char uc = static_cast<unsigned char>(c);

    if (std::isspace(uc)) {
      capitalize_next = true;
    } else if (capitalize_next) {
      c = static_cast<char>(std::toupper(uc));
      capitalize_next = false;
    } else {
      c = static_cast<char>(std::tolower(uc));
    }
  }

  return text;
}

}  // namespace arelto
