// src/main.cpp
#include "game.h"

int main(int argc, char* args[]) {
  rl2::Game game;

  if (game.Initialize()) {
    game.RunGameLoop();
  }

  return 0;
}
