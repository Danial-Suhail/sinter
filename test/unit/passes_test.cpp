/**
 * @file passes_test.cpp
 * @brief Unit tests for graph optimization passes
 */

#include <gtest/gtest.h>
#include "sinter/passes.h"

using namespace sinter;

TEST(FusionPassTest, FusesSimpleAddReluChain) {
    Graph graph;
    graph.add_op(Op(0, OpType::Input, {}, Shape({16, 16})));
    graph.add_op(Op(1, OpType::Input, {}, Shape({16, 16})));
    graph.add_op(Op(2, OpType::Add, {0, 1}, Shape({16, 16})));
    graph.add_op(Op(3, OpType::ReLU, {2}, Shape({16, 16})));
    graph.add_op(Op(4, OpType::Output, {3}, Shape({16, 16})));

    FusionPass pass;
    Graph optimized = pass.optimize(graph);

    EXPECT_EQ(optimized.num_ops(), 4);
    EXPECT_EQ(optimized.get_op(2).type, OpType::FusedOp);
    EXPECT_EQ(optimized.get_op(2).get_attribute("fused_ops"), "add,relu");
    EXPECT_EQ(optimized.get_op(4).input_ids.size(), 1);
    EXPECT_EQ(optimized.get_op(4).input_ids[0], 2);
}

TEST(FusionPassTest, DoesNotFuseWhenAddHasMultipleConsumers) {
    Graph graph;
    graph.add_op(Op(0, OpType::Input, {}, Shape({8, 8})));
    graph.add_op(Op(1, OpType::Input, {}, Shape({8, 8})));
    graph.add_op(Op(2, OpType::Add, {0, 1}, Shape({8, 8})));
    graph.add_op(Op(3, OpType::ReLU, {2}, Shape({8, 8})));
    graph.add_op(Op(4, OpType::Output, {2}, Shape({8, 8})));
    graph.add_op(Op(5, OpType::Output, {3}, Shape({8, 8})));

    FusionPass pass;
    Graph optimized = pass.optimize(graph);

    EXPECT_EQ(optimized.num_ops(), 6);
    EXPECT_EQ(optimized.get_op(2).type, OpType::Add);
    EXPECT_EQ(optimized.get_op(3).type, OpType::ReLU);
    EXPECT_EQ(optimized.get_op(5).input_ids[0], 3);
}
