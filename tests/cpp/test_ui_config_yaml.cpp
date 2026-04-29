// Unit tests for YAML converters backing UI config files.

#include <gtest/gtest.h>

#include "config/ui_config.h"
#include "config/ui_config_yaml.h"
#include "yaml-cpp/yaml.h"

namespace arelto {
namespace {

void ExpectColorEq(const SDL_Color& actual, const SDL_Color& expected) {
  EXPECT_EQ(actual.r, expected.r);
  EXPECT_EQ(actual.g, expected.g);
  EXPECT_EQ(actual.b, expected.b);
  EXPECT_EQ(actual.a, expected.a);
}

TEST(UIConfigYamlTest, UIFontConfigDecodePartialYamlKeepsExistingValues) {
  UIFontConfig config;
  config.font_size_small = 1;
  config.font_size_medium = 2;
  config.font_size_large = 3;
  config.font_size_huge = 4;

  const YAML::Node node = YAML::Load("font_size_small: 18\n");

  ASSERT_TRUE(YAML::convert<UIFontConfig>::decode(node, config));
  EXPECT_EQ(config.font_size_small, 18);
  EXPECT_EQ(config.font_size_medium, 2);
}

TEST(UIConfigYamlTest, UIFontConfigDecodeInvalidFieldKeepsExistingValue) {
  UIFontConfig config;
  config.font_size_medium = 26;

  const YAML::Node node = YAML::Load("font_size_medium: invalid\n");

  ASSERT_TRUE(YAML::convert<UIFontConfig>::decode(node, config));
  EXPECT_EQ(config.font_size_medium, 26);
}

TEST(UIConfigYamlTest, UIFontConfigDecodeNonMapReturnsFalse) {
  UIFontConfig config;

  const YAML::Node node = YAML::Load("- 1\n- 2\n");

  EXPECT_FALSE(YAML::convert<UIFontConfig>::decode(node, config));
}

TEST(UIConfigYamlTest, SDLColorDecodeValidChannelsOverrideExistingColor) {
  SDL_Color color{1, 2, 3, 4};

  const YAML::Node node = YAML::Load(R"(
r: 10
g: 20
b: 30
a: 40
)");

  ASSERT_TRUE(YAML::convert<SDL_Color>::decode(node, color));
  ExpectColorEq(color, SDL_Color{10, 20, 30, 40});
}

TEST(UIConfigYamlTest, SDLColorDecodeOutOfRangeChannelKeepsExistingValue) {
  SDL_Color color{1, 2, 3, 4};

  const YAML::Node node = YAML::Load(R"(
r: 300
g: -1
b: 30
a: 40
)");

  ASSERT_TRUE(YAML::convert<SDL_Color>::decode(node, color));
  ExpectColorEq(color, SDL_Color{1, 2, 30, 40});
}

TEST(UIConfigYamlTest, UIColorsConfigDecodePartialNestedColorKeepsExisting) {
  UIColorsConfig config;
  config.positive_green = SDL_Color{1, 2, 3, 4};
  config.negative_red = SDL_Color{5, 6, 7, 8};

  const YAML::Node node = YAML::Load(R"(
positive_green:
  r: 9
  b: 11
)");

  ASSERT_TRUE(YAML::convert<UIColorsConfig>::decode(node, config));
  ExpectColorEq(config.positive_green, SDL_Color{9, 2, 11, 4});
  ExpectColorEq(config.negative_red, SDL_Color{5, 6, 7, 8});
}

}  // namespace
}  // namespace arelto
