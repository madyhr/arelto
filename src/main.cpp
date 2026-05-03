// src/main.cpp
#include "game.h"

int main(int /*argc*/, char* /*args*/[]) {
  arelto::Game game;

  if (game.Initialize()) {
    game.RunGameLoop();
    game.Shutdown();
  }

  return 0;
}
