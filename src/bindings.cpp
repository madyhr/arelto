// src/bindings.cpp
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include "game.h"

namespace py = pybind11;

PYBIND11_MODULE(rl2_py, m) {
  m.doc() = "RL2 Game Python Bindings";

  py::class_<rl2::Game>(m, "Game")
      .def(py::init())
      .def("initialize", &rl2::Game::Initialize)
      .def("process_input", &rl2::Game::ProcessInput)
      .def("run", &rl2::Game::RunGameLoop)
      .def("step", &rl2::Game::StepGame)
      .def("render", &rl2::Game::RenderGame)
      .def("reset", &rl2::Game::ResetGame)
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

               self.action_manager_.ReadActionBuffer(
                   static_cast<float*>(info.ptr), static_cast<int>(info.size),
                   self.scene_);
             }
           })
      .def("get_action_size",
           [](rl2::Game& self) {
             return self.action_manager_.GetActionSize(self.scene_);
           })
      .def("shutdown", &rl2::Game::Shutdown)
      .def("get_game_state", &rl2::Game::GetGameState);
};
