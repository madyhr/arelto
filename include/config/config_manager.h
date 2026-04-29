#ifndef RL2_CONFIG_CONFIG_MANAGER_H_
#define RL2_CONFIG_CONFIG_MANAGER_H_

#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include "yaml-cpp/yaml.h"

namespace arelto {

class ConfigManager {
 public:
  bool LoadFile(const std::string& config_key,
                const std::filesystem::path& file_path);
  bool HasConfig(const std::string& config_key) const;

  int GetInt(const std::string& config_key, const std::string& field_path,
             int fallback) const;
  float GetFloat(const std::string& config_key, const std::string& field_path,
                 float fallback) const;
  bool GetBool(const std::string& config_key, const std::string& field_path,
               bool fallback) const;
  std::string GetString(const std::string& config_key,
                        const std::string& field_path,
                        const std::string& fallback) const;

  template <typename T>
  T GetStruct(const std::string& config_key, const std::string& field_path,
              const T& fallback) const {
    const YAML::Node node = ResolveNode(config_key, field_path);
    if (!node) {
      return fallback;
    }

    T value = fallback;
    try {
      if (!YAML::convert<T>::decode(node, value)) {
        std::cerr << "Failed to decode struct config at " << config_key;
        if (!field_path.empty()) {
          std::cerr << "." << field_path;
        }
        std::cerr << "; using fallback defaults.\n";
        return fallback;
      }
    } catch (const YAML::Exception& e) {
      std::cerr << "YAML exception while decoding struct config at "
                << config_key;
      if (!field_path.empty()) {
        std::cerr << "." << field_path;
      }
      std::cerr << ": " << e.what() << '\n';
      return fallback;
    }

    return value;
  }

 private:
  std::unordered_map<std::string, YAML::Node> configs_;

  YAML::Node ResolveNode(const std::string& config_key,
                         const std::string& field_path) const;
};

}  // namespace arelto

#endif
