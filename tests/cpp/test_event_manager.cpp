// tests/cpp/test_event_manager.cpp
// Unit tests for EventManager class

#include <gtest/gtest.h>

#include "event_manager.h"
#include "scene.h"
#include "test_helpers.h"

namespace arelto {
namespace {

class EventManagerTest : public ::testing::Test {
 protected:
  void SetUp() override { scene_ = testing::CreateTestScene(); }

  Scene scene_;
  EventManager event_manager_;

  EventContext MakeEventContext() { return EventContext{scene_}; }
};

// =============================================================================
// Queue / Storage Tests
// =============================================================================

TEST_F(EventManagerTest, GetEvents_InitiallyEmpty) {
  EXPECT_TRUE(event_manager_.GetEvents().empty());
}

TEST_F(EventManagerTest, Emit_SingleEvent_Stored) {
  event_manager_.Emit(EnemyKilledEvent{/*enemy_idx=*/0});

  ASSERT_EQ(event_manager_.GetEvents().size(), 1);
  ASSERT_TRUE(
      std::holds_alternative<EnemyKilledEvent>(event_manager_.GetEvents()[0]));

  const auto& e = std::get<EnemyKilledEvent>(event_manager_.GetEvents()[0]);
  EXPECT_EQ(e.enemy_idx, 0);
}

TEST_F(EventManagerTest, Emit_MultipleEvents_AllStored) {
  event_manager_.Emit(EnemyKilledEvent{0});
  event_manager_.Emit(PlayerDamagedEvent{2, 10});
  event_manager_.Emit(ExpGemCollectedEvent{5, 50});

  EXPECT_EQ(event_manager_.GetEvents().size(), 3);
}

TEST_F(EventManagerTest, Flush_ClearsAllEvents) {
  event_manager_.Emit(EnemyKilledEvent{0});
  event_manager_.Emit(PlayerDamagedEvent{2, 10});

  ASSERT_EQ(event_manager_.GetEvents().size(), 2);

  event_manager_.Flush();

  EXPECT_TRUE(event_manager_.GetEvents().empty());
}

TEST_F(EventManagerTest, Flush_AllowsNewEventsAfterFlush) {
  event_manager_.Emit(EnemyKilledEvent{0});
  event_manager_.Flush();

  event_manager_.Emit(ExpGemCollectedEvent{3, 100});

  ASSERT_EQ(event_manager_.GetEvents().size(), 1);
  EXPECT_TRUE(std::holds_alternative<ExpGemCollectedEvent>(
      event_manager_.GetEvents()[0]));
}

// =============================================================================
// Subscribe / Dispatch Tests
// =============================================================================

TEST_F(EventManagerTest, Subscribe_HandlerCalledOnDispatch) {
  bool called = false;
  event_manager_.Subscribe<EnemyKilledEvent>(
      [&](const EnemyKilledEvent&, EventContext&) { called = true; });

  event_manager_.Emit(EnemyKilledEvent{0});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_TRUE(called);
}

TEST_F(EventManagerTest, Subscribe_HandlerNotCalledForOtherType) {
  bool called = false;
  event_manager_.Subscribe<EnemyKilledEvent>(
      [&](const EnemyKilledEvent&, EventContext&) { called = true; });

  event_manager_.Emit(PlayerDamagedEvent{0, 10});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_FALSE(called);
}

TEST_F(EventManagerTest, Dispatch_CallsHandlersInEmitOrder) {
  std::vector<int> order;
  event_manager_.Subscribe<EnemyKilledEvent>(
      [&](const EnemyKilledEvent& e, EventContext&) {
        order.push_back(e.enemy_idx);
      });

  event_manager_.Emit(EnemyKilledEvent{/*enemy_idx=*/1});
  event_manager_.Emit(EnemyKilledEvent{/*enemy_idx=*/2});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  ASSERT_EQ(order.size(), 2);
  EXPECT_EQ(order[0], 1);
  EXPECT_EQ(order[1], 2);
}

TEST_F(EventManagerTest, Dispatch_MultipleHandlersSameType_BothCalled) {
  int call_count = 0;
  event_manager_.Subscribe<ExpGemCollectedEvent>(
      [&](const ExpGemCollectedEvent&, EventContext&) { ++call_count; });
  event_manager_.Subscribe<ExpGemCollectedEvent>(
      [&](const ExpGemCollectedEvent&, EventContext&) { ++call_count; });

  event_manager_.Emit(ExpGemCollectedEvent{0, 50});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_EQ(call_count, 2);
}

TEST_F(EventManagerTest, Dispatch_NoHandlers_NoError) {
  event_manager_.Emit(ChestOpenedEvent{0});
  auto event_context = MakeEventContext();
  EXPECT_NO_FATAL_FAILURE(event_manager_.Dispatch(event_context));
}

TEST_F(EventManagerTest, Dispatch_HandlerReceivesCorrectPayload) {
  PlayerDamagedEvent received{};
  event_manager_.Subscribe<PlayerDamagedEvent>(
      [&](const PlayerDamagedEvent& e, EventContext&) { received = e; });

  event_manager_.Emit(PlayerDamagedEvent{/*enemy_idx=*/3, /*damage=*/15});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_EQ(received.enemy_idx, 3);
  EXPECT_EQ(received.damage, 15);
}

TEST_F(EventManagerTest,
       Dispatch_EnemyDamagedEvent_HandlerReceivesCorrectPayload) {
  EnemyDamagedEvent received{};
  event_manager_.Subscribe<EnemyDamagedEvent>(
      [&](const EnemyDamagedEvent& e, EventContext&) { received = e; });

  event_manager_.Emit(EnemyDamagedEvent{/*enemy_idx=*/2, /*damage=*/25});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_EQ(received.enemy_idx, 2);
  EXPECT_EQ(received.damage, 25);
}

TEST_F(EventManagerTest,
       Dispatch_HandlerEmitsDuringDispatch_CascadingEventDispatched) {
  bool gem_handler_called = false;
  event_manager_.Subscribe<EnemyKilledEvent>([&](const EnemyKilledEvent&,
                                                 EventContext&) {
    event_manager_.Emit(ExpGemCollectedEvent{/*gem_idx=*/9, /*exp_value=*/100});
  });
  event_manager_.Subscribe<ExpGemCollectedEvent>(
      [&](const ExpGemCollectedEvent& e, EventContext&) {
        EXPECT_EQ(e.gem_idx, 9);
        gem_handler_called = true;
      });

  event_manager_.Emit(EnemyKilledEvent{0});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_TRUE(gem_handler_called);
}

TEST_F(EventManagerTest, Dispatch_MultipleEventTypes_AllHandlersFire) {
  bool enemy_handler_called = false;
  bool gem_handler_called = false;
  event_manager_.Subscribe<EnemyKilledEvent>(
      [&](const EnemyKilledEvent&, EventContext&) {
        enemy_handler_called = true;
      });
  event_manager_.Subscribe<ExpGemCollectedEvent>(
      [&](const ExpGemCollectedEvent&, EventContext&) {
        gem_handler_called = true;
      });

  event_manager_.Emit(EnemyKilledEvent{0});
  event_manager_.Emit(ExpGemCollectedEvent{5, 50});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_TRUE(enemy_handler_called);
  EXPECT_TRUE(gem_handler_called);
}

TEST_F(EventManagerTest, Dispatch_ContextMutationVisibleToSubsequentHandlers) {
  int health_seen_by_second_handler = 0;
  event_manager_.Subscribe<EnemyKilledEvent>(
      [&](const EnemyKilledEvent&, EventContext& event_context) {
        event_context.scene.player.stats_.health += 99;
      });
  event_manager_.Subscribe<EnemyKilledEvent>([&](const EnemyKilledEvent&,
                                                 EventContext& event_context) {
    health_seen_by_second_handler = event_context.scene.player.stats_.health;
  });

  int initial_health = scene_.player.stats_.health;
  event_manager_.Emit(EnemyKilledEvent{0});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_EQ(health_seen_by_second_handler, initial_health + 99);
}

TEST_F(EventManagerTest, Dispatch_EmptyQueue_HandlerNotCalled) {
  bool called = false;
  event_manager_.Subscribe<EnemyKilledEvent>(
      [&](const EnemyKilledEvent&, EventContext&) { called = true; });

  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_FALSE(called);
}

// =============================================================================
// Unsubscribe Tests
// =============================================================================

TEST_F(EventManagerTest, Unsubscribe_AfterSubscribe_HandlerNotCalled) {
  bool called = false;
  auto id = event_manager_.Subscribe<EnemyKilledEvent>(
      [&](const EnemyKilledEvent&, EventContext&) { called = true; });

  event_manager_.Unsubscribe<EnemyKilledEvent>(id);

  event_manager_.Emit(EnemyKilledEvent{0});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_FALSE(called);
}

TEST_F(EventManagerTest, Unsubscribe_InvalidId_IsNoOp) {
  bool called = false;
  event_manager_.Subscribe<EnemyKilledEvent>(
      [&](const EnemyKilledEvent&, EventContext&) { called = true; });

  event_manager_.Unsubscribe<EnemyKilledEvent>(9999);

  event_manager_.Emit(EnemyKilledEvent{0});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_TRUE(called);
}

TEST_F(EventManagerTest,
       Unsubscribe_OneOfMultipleHandlers_OnlyRemovedHandlerSkipped) {
  bool called_a = false;
  bool called_b = false;
  auto id_a = event_manager_.Subscribe<EnemyKilledEvent>(
      [&](const EnemyKilledEvent&, EventContext&) { called_a = true; });
  event_manager_.Subscribe<EnemyKilledEvent>(
      [&](const EnemyKilledEvent&, EventContext&) { called_b = true; });

  event_manager_.Unsubscribe<EnemyKilledEvent>(id_a);

  event_manager_.Emit(EnemyKilledEvent{0});
  auto event_context = MakeEventContext();
  event_manager_.Dispatch(event_context);

  EXPECT_FALSE(called_a);
  EXPECT_TRUE(called_b);
}

}  // namespace
}  // namespace arelto
