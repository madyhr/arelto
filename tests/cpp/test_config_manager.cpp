// Unit tests for YAML-backed config loading and typed fallback behavior.

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "config/config_manager.h"

namespace arelto {
namespace config_manager_test {

struct StructConfig {
  int first = 0;
  int second = 0;
};

}  // namespace config_manager_test
}  // namespace arelto

namespace YAML {

template <>
struct convert<arelto::config_manager_test::StructConfig> {
  static Node encode(const arelto::config_manager_test::StructConfig& rhs) {
    Node node;
    node["first"] = rhs.first;
    node["second"] = rhs.second;
    return node;
  }

  static bool decode(const Node& node,
                     arelto::config_manager_test::StructConfig& rhs) {
    if (!node.IsMap()) {
      return false;
    }

    if (node["decode_ok"] && !node["decode_ok"].as<bool>()) {
      return false;
    }

    if (node["force_throw"]) {
      node["force_throw"].as<int>();
    }

    if (node["first"]) {
      rhs.first = node["first"].as<int>();
    }
    if (node["second"]) {
      rhs.second = node["second"].as<int>();
    }
    return true;
  }
};

}  // namespace YAML

namespace arelto {
namespace {

using config_manager_test::StructConfig;

class TempYamlFile {
 public:
  explicit TempYamlFile(const std::string& yaml_contents) {
    const auto now =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("arelto_config_test_" + std::to_string(now) + ".yaml");

    std::ofstream output(path_);
    output << yaml_contents;
  }

  ~TempYamlFile() {
    std::error_code ignore;
    std::filesystem::remove(path_, ignore);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

std::filesystem::path MissingYamlPath() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("arelto_missing_config_test_" + std::to_string(now) + ".yaml");
}

TEST(ConfigManagerTest, LoadFileReturnsTrueAndMarksConfigPresent) {
  TempYamlFile file("value: 42\n");

  ConfigManager manager;

  EXPECT_TRUE(manager.LoadFile("test.load", file.path()));
  EXPECT_TRUE(manager.HasConfig("test.load"));
}

TEST(ConfigManagerTest, MissingFileReturnsFalseAndDoesNotMarkConfigPresent) {
  ConfigManager manager;

  EXPECT_FALSE(manager.LoadFile("test.missing", MissingYamlPath()));
  EXPECT_FALSE(manager.HasConfig("test.missing"));
}

TEST(ConfigManagerTest, MalformedYamlReturnsFalse) {
  TempYamlFile file(R"(
root:
  - [unterminated
)");

  ConfigManager manager;

  EXPECT_FALSE(manager.LoadFile("test.malformed", file.path()));
  EXPECT_FALSE(manager.HasConfig("test.malformed"));
}

TEST(ConfigManagerTest, ScalarGettersReadValuesAtRootAndNestedPath) {
  TempYamlFile file(R"(
int_value: 42
float_value: 1.25
bool_value: true
string_value: "hello"
nested:
  answer: 123
)");

  ConfigManager manager;
  ASSERT_TRUE(manager.LoadFile("test.scalar", file.path()));

  EXPECT_EQ(manager.GetInt("test.scalar", "int_value", -1), 42);
  EXPECT_FLOAT_EQ(manager.GetFloat("test.scalar", "float_value", -1.0f), 1.25f);
  EXPECT_TRUE(manager.GetBool("test.scalar", "bool_value", false));
  EXPECT_EQ(manager.GetString("test.scalar", "string_value", "fallback"),
            "hello");
  EXPECT_EQ(manager.GetInt("test.scalar", "nested.answer", -1), 123);
}

TEST(ConfigManagerTest, ScalarGettersMissingLoadedFieldUsesFallback) {
  TempYamlFile file("present: 42\n");

  ConfigManager manager;
  ASSERT_TRUE(manager.LoadFile("test.loaded", file.path()));

  EXPECT_EQ(manager.GetInt("test.loaded", "missing", 99), 99);
}

TEST(ConfigManagerTest, ScalarGettersInvalidScalarUsesFallback) {
  TempYamlFile file("int_value: not_an_int\n");

  ConfigManager manager;
  ASSERT_TRUE(manager.LoadFile("test.invalid", file.path()));

  EXPECT_EQ(manager.GetInt("test.invalid", "int_value", 7), 7);
}

TEST(ConfigManagerTest, ScalarGettersNullValueUsesFallback) {
  TempYamlFile file("int_value: null\n");

  ConfigManager manager;
  ASSERT_TRUE(manager.LoadFile("test.null", file.path()));

  EXPECT_EQ(manager.GetInt("test.null", "int_value", 7), 7);
}

TEST(ConfigManagerTest, PathResolutionInvalidPathUsesFallback) {
  TempYamlFile file(R"(
scalar: 42
nested:
  answer: 123
)");

  ConfigManager manager;
  ASSERT_TRUE(manager.LoadFile("test.paths", file.path()));

  EXPECT_EQ(manager.GetInt("test.paths", "nested..answer", 7), 7);
  EXPECT_EQ(manager.GetInt("test.paths", "scalar.child", 8), 8);
}

TEST(ConfigManagerTest, GetStructMergesYamlOverCallerFallback) {
  TempYamlFile file("first: 42\n");

  ConfigManager manager;
  ASSERT_TRUE(manager.LoadFile("test.struct", file.path()));

  const StructConfig fallback{10, 20};
  const StructConfig loaded =
      manager.GetStruct<StructConfig>("test.struct", "", fallback);

  EXPECT_EQ(loaded.first, 42);
  EXPECT_EQ(loaded.second, 20);
}

TEST(ConfigManagerTest, GetStructMissingConfigOrPathReturnsFallback) {
  TempYamlFile file("nested: {}\n");

  ConfigManager manager;
  ASSERT_TRUE(manager.LoadFile("test.struct", file.path()));

  const StructConfig fallback{10, 20};

  StructConfig missing_config =
      manager.GetStruct<StructConfig>("missing.config", "", fallback);
  EXPECT_EQ(missing_config.first, 10);
  EXPECT_EQ(missing_config.second, 20);

  StructConfig missing_path = manager.GetStruct<StructConfig>(
      "test.struct", "nested.missing", fallback);
  EXPECT_EQ(missing_path.first, 10);
  EXPECT_EQ(missing_path.second, 20);
}

TEST(ConfigManagerTest, GetStructDecodeReturnsFalseReturnsFallback) {
  TempYamlFile file("decode_ok: false\nfirst: 42\n");

  ConfigManager manager;
  ASSERT_TRUE(manager.LoadFile("test.struct", file.path()));

  const StructConfig fallback{10, 20};
  const StructConfig loaded =
      manager.GetStruct<StructConfig>("test.struct", "", fallback);

  EXPECT_EQ(loaded.first, 10);
  EXPECT_EQ(loaded.second, 20);
}

TEST(ConfigManagerTest, GetStructDecodeExceptionReturnsFallback) {
  TempYamlFile file("force_throw: not_an_int\nfirst: 42\n");

  ConfigManager manager;
  ASSERT_TRUE(manager.LoadFile("test.struct", file.path()));

  const StructConfig fallback{10, 20};
  const StructConfig loaded =
      manager.GetStruct<StructConfig>("test.struct", "", fallback);

  EXPECT_EQ(loaded.first, 10);
  EXPECT_EQ(loaded.second, 20);
}

TEST(ConfigManagerTest, FailedReloadLeavesExistingConfig) {
  TempYamlFile file("value: 42\n");

  ConfigManager manager;
  ASSERT_TRUE(manager.LoadFile("test.reload", file.path()));

  EXPECT_FALSE(manager.LoadFile("test.reload", MissingYamlPath()));
  EXPECT_TRUE(manager.HasConfig("test.reload"));
  EXPECT_EQ(manager.GetInt("test.reload", "value", -1), 42);
}

}  // namespace
}  // namespace arelto
