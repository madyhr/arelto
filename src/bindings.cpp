// src/bindings.cpp
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include "game.h"

namespace py = pybind11;

PYBIND11_MODULE(rl2_py, m) {
  m.doc() = "RL2 Game Python Bindings";

  py::class_<rl2::Game>(m, "Game") 
    .def(py::init())
    .def("initialize", &rl2::Game::Initialize)
    .def("run", &rl2::Game::RunGameLoop)
    .def("shutdown", &rl2::Game::Shutdown);

};
