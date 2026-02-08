// src/audio_manager.cpp
#include "audio_manager.h"
#include <SDL_mixer.h>
#include <algorithm>
#include "constants/audio_manager.h"

namespace arelto {

AudioManager::AudioManager() {};
AudioManager::~AudioManager() {
  Shutdown();
};

bool AudioManager::Initialize() {

  if (Mix_Init(MIX_INIT_MP3) == 0) {
    printf("SDL_mixer could not initialize MP3 support! Error: %s\n",
           Mix_GetError());
    return false;
  }

  if (Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) < 0) {
    printf("SDL_mixer could not initialize! SDL_mixer Error: %s\n",
           Mix_GetError());
    return false;
  }

  soundtrack_ = Mix_LoadMUS("assets/music/arelto_a6ee808.mp3");
  if (soundtrack_ == nullptr) {
    printf("Failed to load music! SDL_mixer Error: %s\n", Mix_GetError());
  }

  SetMusicVolume(kMusicVolumeInitial);

  return true;
};

void AudioManager::StopMusic() {
  Mix_HaltMusic();
};

void AudioManager::PlayMusic() {
  // If music is already playing, we do not want to abruptly restart it.
  if (Mix_PlayingMusic()) {
    return;
  }

  Mix_FadeInMusic(soundtrack_, -1, kMusicVolumeFadeInMs);
}

void AudioManager::SetMusicVolume(float volume) {

  if (is_music_muted) {
    is_music_muted = false;
  }

  music_volume_ = std::clamp(volume, 0.0f, 1.0f);

  // SDL Mixer Music Volume goes from 0 to 128.
  int volume_int = static_cast<int>(music_volume_ * 128);
  Mix_VolumeMusic(volume_int);
};

float AudioManager::GetMusicVolume() {
  return music_volume_;
};

void AudioManager::ToggleMusic() {

  if (is_music_muted) {
    SetMusicVolume(music_volume_);
    return;
  }

  // Bypass the music volume setter to avoid overwriting the music_volume_ attribute.
  Mix_VolumeMusic(0);
  is_music_muted = true;
};

void AudioManager::IncreaseMusicVolume() {
  SetMusicVolume(music_volume_ + kMusicVolumeStep);
};

void AudioManager::DecreaseMusicVolume() {
  SetMusicVolume(music_volume_ - kMusicVolumeStep);
};

void AudioManager::Shutdown() {
  Mix_FreeMusic(soundtrack_);
  soundtrack_ = nullptr;
  Mix_Quit();
};

}  // namespace arelto
