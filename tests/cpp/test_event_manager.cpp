// tests/cpp/test_event_manager.cpp
// Unit tests for EventManager class

#include <gtest/gtest.h>

#include "event_manager.h"

namespace arelto {
namespace {

class EventManagerTest : public ::testing::Test {
 protected:
  EventManager event_manager_;
};

// =============================================================================
// Core API Tests
// =============================================================================

TEST_F(EventManagerTest, GetEvents_InitiallyEmpty) {
  EXPECT_TRUE(event_manager_.GetEvents().empty());
}

TEST_F(EventManagerTest, Emit_SingleEvent_Stored) {
  event_manager_.Emit({EventType::enemy_killed, /*entity_index=*/0,
                       /*secondary_index=*/1,
                       /*value=*/25.0f});

  ASSERT_EQ(event_manager_.GetEvents().size(), 1);

  const GameEvent& event = event_manager_.GetEvents()[0];
  EXPECT_EQ(event.type, EventType::enemy_killed);
  EXPECT_EQ(event.entity_index, 0);
  EXPECT_EQ(event.secondary_index, 1);
  EXPECT_FLOAT_EQ(event.value, 25.0f);
}

TEST_F(EventManagerTest, Emit_MultipleEvents_AllStored) {
  event_manager_.Emit({EventType::enemy_killed, 0, 1, 25.0f});
  event_manager_.Emit({EventType::player_damaged, 2, -1, 10.0f});
  event_manager_.Emit({EventType::gem_collected, 5, -1, 50.0f});

  EXPECT_EQ(event_manager_.GetEvents().size(), 3);
}

TEST_F(EventManagerTest, Flush_ClearsAllEvents) {
  event_manager_.Emit({EventType::enemy_killed, 0, 1, 25.0f});
  event_manager_.Emit({EventType::player_damaged, 2, -1, 10.0f});

  ASSERT_EQ(event_manager_.GetEvents().size(), 2);

  event_manager_.Flush();

  EXPECT_TRUE(event_manager_.GetEvents().empty());
}

TEST_F(EventManagerTest, Flush_AllowsNewEventsAfterFlush) {
  event_manager_.Emit({EventType::enemy_killed, 0, 1, 25.0f});
  event_manager_.Flush();

  event_manager_.Emit({EventType::gem_collected, 3, -1, 100.0f});

  ASSERT_EQ(event_manager_.GetEvents().size(), 1);
  EXPECT_EQ(event_manager_.GetEvents()[0].type, EventType::gem_collected);
}

// =============================================================================
// Filtered Access Tests
// =============================================================================

TEST_F(EventManagerTest, GetEventsFiltered_ReturnsOnlyMatchingType) {
  event_manager_.Emit({EventType::enemy_killed, 0, 1, 25.0f});
  event_manager_.Emit({EventType::player_damaged, 2, -1, 10.0f});
  event_manager_.Emit({EventType::enemy_killed, 3, 4, 30.0f});
  event_manager_.Emit({EventType::gem_collected, 5, -1, 50.0f});

  auto kills = event_manager_.GetEvents(EventType::enemy_killed);
  ASSERT_EQ(kills.size(), 2);
  EXPECT_EQ(kills[0].entity_index, 0);
  EXPECT_EQ(kills[1].entity_index, 3);

  auto damage = event_manager_.GetEvents(EventType::player_damaged);
  ASSERT_EQ(damage.size(), 1);
  EXPECT_EQ(damage[0].entity_index, 2);
}

TEST_F(EventManagerTest, GetEventsFiltered_ReturnsEmptyWhenNoMatch) {
  event_manager_.Emit({EventType::enemy_killed, 0, 1, 25.0f});

  auto gems = event_manager_.GetEvents(EventType::gem_collected);
  EXPECT_TRUE(gems.empty());
}

}  // namespace
}  // namespace arelto
