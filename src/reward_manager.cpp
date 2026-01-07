// src/reward_manager.cpp
#include "reward_manager.h"
#include <array>
#include <stdexcept>
#include "constants/enemy.h"
#include "scene.h"

namespace rl2 {

bool RewardManager::Initialize() {
  terms_.clear();
  RegisterRewardTerms();
  ResetRewardTerms();

  return true;
};

void RewardManager::ResetRewardTerms() {
  for (RewardTerm& term : terms_) {
    term.last_signal.fill(0.0f);
    term.last_value.fill(0.0f);
    term.accumulated_signal.fill(0.0f);
  }
};

// This function is called inside the physics loop to update reward terms
// based on the accumulated changes that happens during the N physics steps
// per game step instead of just the signal of the last physics step.
void RewardManager::UpdateRewardTerms(const Scene& scene) {
  for (RewardTerm& term : terms_) {
    term.last_signal = term.func(scene);

    for (int i = 0; i < kNumEnemies; ++i) {
      term.accumulated_signal[i] += term.last_signal[i];
    }
  }
};

void RewardManager::AddTerm(std::string name, float weight,
                            RewardFunction func) {
  terms_.push_back(RewardTerm{name, weight, func});
};

std::array<float, kNumEnemies> RewardManager::CalculateTotalReward(
    const Scene& scene) {
  std::array<float, kNumEnemies> total_reward{};

  for (RewardTerm& term : terms_) {
    for (int i = 0; i < kNumEnemies; ++i) {
      float value = term.accumulated_signal[i] * term.weight;
      term.last_value[i] = value;
      total_reward[i] += value;
    }
    term.accumulated_signal.fill(0.0f);
  }

  return total_reward;
};

std::map<std::string, std::array<float, kNumEnemies>>
RewardManager::GetLastRewardDict() {
  std::map<std::string, std::array<float, kNumEnemies>> reward_dict;

  for (RewardTerm& term : terms_) {
    reward_dict[term.name] = term.last_value;
  }

  return reward_dict;
};

std::map<std::string, std::array<float, kNumEnemies>>
RewardManager::GetLastSignalDict() {
  std::map<std::string, std::array<float, kNumEnemies>> signal_dict;

  for (RewardTerm& term : terms_) {
    signal_dict[term.name] = term.last_signal;
  }

  return signal_dict;
};

int RewardManager::GetRewardSize() {
  return kNumEnemies;
};

void RewardManager::FillRewardBuffer(float* buffer_ptr, int buffer_size,
                                     const Scene& scene) {
  if (buffer_size != GetRewardSize()) {
    throw std::runtime_error("Buffer size mismatch");
  };

  int idx = 0;
  std::array<float, kNumEnemies> total_rewards = CalculateTotalReward(scene);

  for (int i = 0; i < kNumEnemies; ++i) {
    buffer_ptr[idx++] = total_rewards[i];
  }

  return;
};

}  // namespace rl2
