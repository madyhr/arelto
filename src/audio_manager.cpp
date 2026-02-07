// src/audio_manager.cpp
#include "audio_manager.h"
#include <SDL_mixer.h>

namespace arelto {

AudioManager::AudioManager() {};
AudioManager::~AudioManager() {
  Shutdown();
};

bool AudioManager::Initialize() {
  if (Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) < 0) {
    printf("SDL_mixer could not initialize! SDL_mixer Error: %s\n",
           Mix_GetError());
    return false;
  }

  soundtrack_ = Mix_LoadMUS("assets/music/arelto_a6ee808.mp3");
  if (soundtrack_ == nullptr) {
    printf("Failed to load music! SDL_mixer Error: %s\n", Mix_GetError());
  }

  SetMusicVolume(42);

  return true;
};

void AudioManager::StopMusic() {
  Mix_HaltMusic();
};

void AudioManager::PlayMusic() {
  // If music is already playing, we do not want to abruptly restart it.
  if (!(Mix_PlayingMusic() == 0)) {
    return;
  }

  Mix_PlayMusic(soundtrack_, -1);
}

void AudioManager::SetMusicVolume(int volume) {
  // Volume goes from 0 to 128.
  Mix_VolumeMusic(volume);
};

int AudioManager::GetMusicVolume() {
  return music_volume_;
};

void AudioManager::Shutdown() {
  Mix_FreeMusic(soundtrack_);
  soundtrack_ = nullptr;
  Mix_Quit();
};

}  // namespace arelto
