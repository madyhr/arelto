// src/bindings.cpp
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include "constants.h"
#include "game.h"

namespace py = pybind11;

PYBIND11_MODULE(rl2_py, m) {
  m.doc() = "RL2 Game Python Bindings";

  py::class_<rl2::Game>(m, "Game")
      .def_readonly_static("num_enemies", &rl2::kNumEnemies)
      .def(py::init())
      .def("initialize", &rl2::Game::Initialize)
      .def("process_input", &rl2::Game::ProcessInput)
      .def("run", &rl2::Game::RunGameLoop)
      .def("step", &rl2::Game::StepGame)
      .def("render", &rl2::Game::RenderGame)
      .def("reset", &rl2::Game::ResetGame)
      .def("fill_terminated_buffer",
           [](rl2::Game& self, py::array_t<bool> buffer) {
             py::buffer_info info = buffer.request();

             if (info.ndim != 1) {
               throw std::runtime_error("Terminated buffer must be 1D array");
             }

             bool* info_ptr = static_cast<bool*>(info.ptr);

             for (int i = 0; i < rl2::kNumEnemies; ++i) {
               info_ptr[i] = self.scene_.enemy.is_terminated_latched[i];
               self.scene_.enemy.is_terminated_latched[i] = false;
             }
           })
      .def("fill_truncated_buffer",
           [](rl2::Game& self, py::array_t<bool> buffer) {
             py::buffer_info info = buffer.request();

             if (info.ndim != 1) {
               throw std::runtime_error("Truncated buffer must be 1D array");
             }

             bool* info_ptr = static_cast<bool*>(info.ptr);

             for (int i = 0; i < rl2::kNumEnemies; ++i) {
               info_ptr[i] = self.scene_.enemy.is_truncated_latched[i];
               self.scene_.enemy.is_truncated_latched[i] = false;
             }
           })
      .def("fill_observation_buffer",
           [](rl2::Game& self, py::array_t<float> buffer) {
             py::buffer_info info = buffer.request();

             if (info.ndim != 1) {
               throw std::runtime_error("Observation buffer must be 1D array");
             }

             self.obs_manager_.FillObservationBuffer(
                 static_cast<float*>(info.ptr), static_cast<int>(info.size),
                 self.scene_);
           })
      .def("get_observation_size",
           [](rl2::Game& self) {
             return self.obs_manager_.GetObservationSize(self.scene_);
           })
      .def("apply_action",
           [](rl2::Game& self, py::array_t<float> buffer) {
             py::buffer_info info = buffer.request();

             if (info.ndim != 1) {
               throw std::runtime_error("Action buffer must be 1D array");
             }

             self.action_manager_.ReadActionBuffer(
                 static_cast<float*>(info.ptr), static_cast<int>(info.size),
                 self.scene_);
           })
      .def("get_action_size",
           [](rl2::Game& self) {
             return self.action_manager_.GetActionSize(self.scene_);
           })
      .def("fill_reward_buffer",
           [](rl2::Game& self, py::array_t<float> buffer) {
             py::buffer_info info = buffer.request();

             if (info.ndim != 1) {
               throw std::runtime_error("Reward buffer must be 1D array");
             }

             self.reward_manager_.FillRewardBuffer(
                 static_cast<float*>(info.ptr), static_cast<int>(info.size),
                 self.scene_);
           })
      .def("get_reward_size",
           [](rl2::Game& self) { return self.reward_manager_.GetRewardSize(); })
      .def("shutdown", &rl2::Game::Shutdown)
      .def("get_game_state", &rl2::Game::GetGameState);
};
