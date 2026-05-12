#ifndef RL2_CONFIG_CONFIG_DECODING_H_
#define RL2_CONFIG_CONFIG_DECODING_H_

#include <iostream>
#include <type_traits>
#include "yaml-cpp/yaml.h"

namespace arelto::config::detail {

template <typename T>
struct FieldDecoder {
  static void Decode(const YAML::Node& node, const char* key, T& out,
                     const char* owner) {
    const YAML::Node field = node[key];
    if (!field) {
      return;
    }

    try {
      out = field.as<T>();
    } catch (const YAML::Exception& e) {
      std::cerr << "Invalid config value for " << owner << "." << key << ": "
                << e.what() << '\n';
    }
  }
};

template <typename T>
void DecodeField(const YAML::Node& node, const char* key, T& out,
                 const char* owner) {
  FieldDecoder<std::remove_cv_t<T>>::Decode(node, key, out, owner);
};

template <typename StructType, typename MemberType>
void DecodeMember(const YAML::Node& node, const char* key, StructType& out,
                  MemberType StructType::* member, const char* owner) {
  DecodeField(node, key, out.*member, owner);
}

}  // namespace arelto::config::detail

#endif
