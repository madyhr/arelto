#include "config/config_manager.h"

namespace arelto {

namespace {

template <typename T>
T NodeAsOrFallback(const YAML::Node& node, const std::string& config_key,
                   const std::string& field_path, const T& fallback) {
  if (!node.IsDefined() || node.IsNull()) {
    return fallback;
  }

  try {
    return node.as<T>();
  } catch (const YAML::Exception& e) {
    std::cerr << "Invalid scalar config value at " << config_key;
    if (!field_path.empty()) {
      std::cerr << "." << field_path;
    }
    std::cerr << ": " << e.what() << "; using fallback.\n";
    return fallback;
  }
}

}  // namespace

bool ConfigManager::LoadFile(const std::string& config_key,
                             const std::filesystem::path& file_path) {
  try {
    configs_[config_key] = YAML::LoadFile(file_path.string());
    return true;
  } catch (const YAML::Exception& e) {
    std::cerr << "Failed to load YAML config file '" << file_path.string()
              << "' (" << config_key << "): " << e.what()
              << "; using fallback defaults.\n";
    return false;
  }
}

bool ConfigManager::HasConfig(const std::string& config_key) const {
  return configs_.find(config_key) != configs_.end();
}

int ConfigManager::GetInt(const std::string& config_key,
                          const std::string& field_path, int fallback) const {
  return NodeAsOrFallback<int>(ResolveNode(config_key, field_path), config_key,
                               field_path, fallback);
}

float ConfigManager::GetFloat(const std::string& config_key,
                              const std::string& field_path,
                              float fallback) const {
  return NodeAsOrFallback<float>(ResolveNode(config_key, field_path),
                                 config_key, field_path, fallback);
}

bool ConfigManager::GetBool(const std::string& config_key,
                            const std::string& field_path,
                            bool fallback) const {
  return NodeAsOrFallback<bool>(ResolveNode(config_key, field_path), config_key,
                                field_path, fallback);
}

std::string ConfigManager::GetString(const std::string& config_key,
                                     const std::string& field_path,
                                     const std::string& fallback) const {
  return NodeAsOrFallback<std::string>(ResolveNode(config_key, field_path),
                                       config_key, field_path, fallback);
}

// Returns the resolved `YAML::Node` based on the given key and path.
// Implementation notes:
// 1. Since dot-paths are used to traverse the node, you cannot use the `.`
// character in the key itself.
YAML::Node ConfigManager::ResolveNode(const std::string& config_key,
                                      const std::string& field_path) const {
  const auto root_it = configs_.find(config_key);
  if (root_it == configs_.end()) {
    return YAML::Node();
  }

  YAML::Node current = root_it->second;
  if (field_path.empty()) {
    return current;
  }

  std::size_t start = 0;
  while (start < field_path.size()) {
    const std::size_t dot = field_path.find('.', start);
    const std::string token = field_path.substr(
        start, dot == std::string::npos ? std::string::npos : dot - start);

    if (token.empty() || !current.IsMap()) {
      return YAML::Node();
    }

    const YAML::Node next = current[token];
    if (!next.IsDefined()) {
      return YAML::Node();
    }
    current.reset(next);

    if (dot == std::string::npos) {
      break;
    }
    start = dot + 1;
  }

  return current;
}

}  // namespace arelto
