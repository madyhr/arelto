// include/audio_manager.h
#ifndef RL2_AUDIO_MANAGER_H_
#define RL2_AUDIO_MANAGER_H_

#include <SDL_mixer.h>
namespace arelto {

class AudioManager {

 public:
  AudioManager();
  ~AudioManager();

  bool Initialize();
  void Shutdown();

  void StopMusic();
  void PlayMusic();
  void IncreaseMusicVolume();
  void DecreaseMusicVolume();
  void SetMusicVolume(float volume);
  float GetMusicVolume();
  void ToggleMusic();

 private:
  Mix_Music* soundtrack_ = nullptr;
  // Music volume is a value between 0 and 1.
  float music_volume_;
  bool is_music_muted = false;
};

}  // namespace arelto
#endif
