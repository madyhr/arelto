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
  void SetMusicVolume(int volume);
  int GetMusicVolume();

 private:
  Mix_Music* soundtrack_ = nullptr;
  int music_volume_;
};

}  // namespace arelto
#endif
