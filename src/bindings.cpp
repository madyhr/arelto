// src/bindings.cpp
#include <pybind11/detail/common.h>
#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include "constants/enemy.h"
#include "constants/ray_caster.h"
#include "game.h"

namespace py = pybind11;

PYBIND11_MODULE(arelto_py, m) {
  m.doc() = "Arelto Game Python Bindings";

  py::native_enum<arelto::GameState>(m, "GameState", "enum.IntEnum")
      .value("IN_START_SCREEN", arelto::GameState::in_start_screen)
      .value("IN_MAIN_MENU", arelto::GameState::in_main_menu)
      .value("IS_RUNNING", arelto::GameState::is_running)
      .value("IS_GAMEOVER", arelto::GameState::is_gameover)
      .value("IN_SHUTDOWN", arelto::GameState::in_shutdown)
      .value("IN_SETTINGS_MENU", arelto::GameState::in_settings_menu)
      .value("IN_LEVEL_UP", arelto::GameState::in_level_up)
      .value("IN_QUIT_CONFIRM", arelto::GameState::in_quit_confirm)
      .value("IN_CHEST_OPENING", arelto::GameState::in_chest_opening)
      .value("IN_ITEM_SELECTION", arelto::GameState::in_item_selection)
      .finalize();

  py::class_<arelto::Game>(m, "Game")
      .def_readonly_static("num_enemies", &arelto::kNumEnemies)
      .def(py::init())
      .def("initialize", &arelto::Game::Initialize)
      .def("process_input", &arelto::Game::ProcessInput)
      .def("run", &arelto::Game::RunGameLoop)
      .def("step", &arelto::Game::StepGame)
      .def("render", &arelto::Game::RenderGame)
      .def("reset", &arelto::Game::ResetGame)
      .def("fill_terminated_buffer",
           [](arelto::Game& self, const py::array_t<bool>& buffer) {
             py::buffer_info info = buffer.request();

             if (info.ndim != 1) {
               throw std::runtime_error("Terminated buffer must be 1D array");
             }

             bool* info_ptr = static_cast<bool*>(info.ptr);

             for (int i = 0; i < arelto::kNumEnemies; ++i) {
               info_ptr[i] = self.scene_.enemy.is_terminated_latched[i];
               self.scene_.enemy.is_terminated_latched[i] = false;
             }
           })
      .def("fill_truncated_buffer",
           [](arelto::Game& self, const py::array_t<bool>& buffer) {
             py::buffer_info info = buffer.request();

             if (info.ndim != 1) {
               throw std::runtime_error("Truncated buffer must be 1D array");
             }

             bool* info_ptr = static_cast<bool*>(info.ptr);

             for (int i = 0; i < arelto::kNumEnemies; ++i) {
               info_ptr[i] = self.scene_.enemy.is_truncated_latched[i];
               self.scene_.enemy.is_truncated_latched[i] = false;
             }
           })
      .def("fill_observation_buffer",
           [](arelto::Game& self, const py::array_t<float>& buffer) {
             py::buffer_info info = buffer.request();

             if (info.ndim != 2) {
               throw std::runtime_error("Observation buffer must be 2D array");
             }

             self.obs_manager_.FillObservationBuffer(
                 static_cast<float*>(info.ptr), static_cast<int>(info.size),
                 self.scene_);
           })
      .def("get_observation_size",
           [](arelto::Game& self) {
             return self.obs_manager_.GetObservationSize();
           })
      .def("apply_action",
           [](arelto::Game& self, const py::array_t<int>& buffer) {
             py::buffer_info info = buffer.request();

             if (info.ndim != 2) {
               throw std::runtime_error("Action buffer must be 2D array");
             }

             self.action_manager_.ReadActionBuffer(static_cast<int*>(info.ptr),
                                                   static_cast<int>(info.size),
                                                   self.scene_);
           })
      .def("get_action_size",
           [](arelto::Game& self) {
             return self.action_manager_.GetActionSize();
           })
      .def("fill_reward_buffer",
           [](arelto::Game& self, const py::array_t<float>& buffer) {
             py::buffer_info info = buffer.request();

             if (info.ndim != 1) {
               throw std::runtime_error("Reward buffer must be 1D array");
             }

             self.reward_manager_.FillRewardBuffer(
                 static_cast<float*>(info.ptr), static_cast<int>(info.size));
           })
      .def("get_reward_size",
           [](arelto::Game& self) {
             return self.reward_manager_.GetRewardSize();
           })
      .def("get_enemy_num_rays",
           [](arelto::Game& /*self*/) { return arelto::kNumRays; })
      .def("get_enemy_ray_history_length",
           [](arelto::Game& /*self*/) { return arelto::kRayHistoryLength; })
      .def("shutdown", &arelto::Game::Shutdown)
      .def("set_game_state", &arelto::Game::SetGameState)
      .def("get_game_state", &arelto::Game::GetGameState);
};
