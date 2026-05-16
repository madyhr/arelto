#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "config/item_config.h"
#include "config/item_config_yaml.h"
#include "items.h"

namespace arelto {
namespace {

class TempYamlFile {
 public:
  explicit TempYamlFile(const std::string& yaml_contents) {
    const auto now =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("arelto_items_test_" + std::to_string(now) + ".yaml");

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
         ("arelto_missing_items_test_" + std::to_string(now) + ".yaml");
}

TEST(ItemArchiveTest, MissingItemConfigUsesCompiledDefaults) {
  ItemArchive archive(MissingYamlPath());

  EXPECT_EQ(archive.GetItem(ItemId::elia_armor_plate).flavor_text,
            ItemFlavorText.at(ItemId::elia_armor_plate));
  EXPECT_EQ(archive.GetItem(ItemId::damodei_claw).flavor_text,
            ItemFlavorText.at(ItemId::damodei_claw));
}

TEST(ItemArchiveTest, ItemConfigOverridesMatchingItemFlavorText) {
  TempYamlFile file(R"(
items:
  damodei_claw:
    flavor_text: "Configured claw text."
)");

  ItemArchive archive(file.path());

  EXPECT_EQ(archive.GetItem(ItemId::damodei_claw).flavor_text,
            "Configured claw text.");
}

}  // namespace
}  // namespace arelto
