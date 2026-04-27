/**
 * @file ir_test.cpp
 * @brief Unit tests for IR (graph, parser, printer, validator)
 */

#include <gtest/gtest.h>
#include "sinter/ir.h"

using namespace sinter;

// ============================================================================
// Fixture for IR tests
// ============================================================================

class IRTest : public ::testing::Test {
protected:
    Graph graph;
};

// ============================================================================
// Shape Tests
// ============================================================================

TEST(ShapeTest, NumElements) {
    Shape s1({2, 3, 4});
    EXPECT_EQ(s1.numel(), 24);

    Shape s2({1});
    EXPECT_EQ(s2.numel(), 1);

    Shape s3;
    EXPECT_EQ(s3.numel(), 1);
}

TEST(ShapeTest, StringRepresentation) {
    Shape s({16, 256, 512});
    EXPECT_EQ(s.str(), "[16,256,512]");
}

TEST(ShapeTest, Equality) {
    Shape s1({2, 3, 4});
    Shape s2({2, 3, 4});
    Shape s3({2, 3, 5});

    EXPECT_EQ(s1, s2);
    EXPECT_NE(s1, s3);
}

// ============================================================================
// Op Tests
// ============================================================================

TEST(OpTest, Creation) {
    Op op(0, OpType::MatMul, {1, 2}, Shape({16, 16}));
    EXPECT_EQ(op.id, 0);
    EXPECT_EQ(op.type, OpType::MatMul);
    EXPECT_EQ(op.input_ids.size(), 2);
    EXPECT_EQ(op.output_shape.numel(), 256);
}

TEST(OpTest, Attributes) {
    Op op(0, OpType::MatMul, {}, Shape());
    op.add_attribute("tile_m", "16");
    op.add_attribute("tile_n", "16");

    EXPECT_EQ(op.get_attribute("tile_m"), "16");
    EXPECT_EQ(op.get_attribute("tile_n"), "16");
    EXPECT_EQ(op.get_attribute("nonexistent"), "");
}

TEST(OpTest, OpTypeString) {
    Op op_matmul(0, OpType::MatMul, {}, Shape());
    EXPECT_EQ(op_matmul.op_type_str(), "matmul");

    Op op_add(1, OpType::Add, {}, Shape());
    EXPECT_EQ(op_add.op_type_str(), "add");

    Op op_gelu(2, OpType::GELU, {}, Shape());
    EXPECT_EQ(op_gelu.op_type_str(), "gelu");
}

// ============================================================================
// Graph Tests
// ============================================================================

TEST(GraphTest, AddAndRetrieveOp) {
    Graph g;
    Op op(0, OpType::Input, {}, Shape({16, 16}));
    g.add_op(op);

    const auto& retrieved = g.get_op(0);
    EXPECT_EQ(retrieved.id, 0);
    EXPECT_EQ(retrieved.output_shape.numel(), 256);
}

TEST(GraphTest, TopologicalSort) {
    Graph g;

    Op op0(0, OpType::Input, {}, Shape({2, 2}));
    Op op1(1, OpType::Add, {0, 0}, Shape({2, 2}));
    Op op2(2, OpType::GELU, {1}, Shape({2, 2}));

    g.add_op(op0);
    g.add_op(op1);
    g.add_op(op2);

    auto topo = g.topological_sort();
    EXPECT_EQ(topo.size(), 3);
    EXPECT_EQ(topo[0], 0);
    EXPECT_EQ(topo[1], 1);
    EXPECT_EQ(topo[2], 2);
}

TEST(GraphTest, CycleDetection) {
    Graph g;

    // Create a cycle: 0 -> 1 -> 0
    Op op0(0, OpType::Input, {1}, Shape({2, 2}));
    Op op1(1, OpType::Add, {0, 0}, Shape({2, 2}));

    g.add_op(op0);
    g.add_op(op1);

    EXPECT_THROW(g.topological_sort(), std::invalid_argument);
}

TEST(GraphTest, NumOps) {
    Graph g;
    Op op0(0, OpType::Input, {}, Shape({2, 2}));
    Op op1(1, OpType::Add, {0, 0}, Shape({2, 2}));

    g.add_op(op0);
    g.add_op(op1);

    EXPECT_EQ(g.num_ops(), 2);
}

// ============================================================================
// Parser Tests
// ============================================================================

TEST(ParserTest, SimpleMatMul) {
    std::string ir = R"(
Op 0: input
  shape: [16,16,16]

Op 1: input
  shape: [16,16]

Op 2: matmul
  inputs: [0,1]
  shape: [16,16]
)";

    Graph g = Parser::parse(ir);
    EXPECT_EQ(g.num_ops(), 3);

    const auto& op0 = g.get_op(0);
    EXPECT_EQ(op0.type, OpType::Input);
    EXPECT_EQ(op0.output_shape.str(), "[16,16,16]");

    const auto& op2 = g.get_op(2);
    EXPECT_EQ(op2.type, OpType::MatMul);
    EXPECT_EQ(op2.input_ids.size(), 2);
}

TEST(ParserTest, WithAttributes) {
    std::string ir = R"(
Op 0: input
  shape: [128,256]

Op 1: matmul
  inputs: [0]
  shape: [128,128]
  attr: tile_m=16
  attr: tile_n=16
)";

    Graph g = Parser::parse(ir);
    const auto& op1 = g.get_op(1);
    EXPECT_EQ(op1.get_attribute("tile_m"), "16");
    EXPECT_EQ(op1.get_attribute("tile_n"), "16");
}

TEST(ParserTest, AllOpTypes) {
    std::string ir = R"(
Op 0: input
  shape: [32]

Op 1: add
  inputs: [0,0]
  shape: [32]

Op 2: softmax
  inputs: [1]
  shape: [32]

Op 3: layernorm
  inputs: [2]
  shape: [32]

Op 4: gelu
  inputs: [3]
  shape: [32]

Op 5: relu
  inputs: [4]
  shape: [32]
)";

    Graph g = Parser::parse(ir);
    EXPECT_EQ(g.num_ops(), 6);
    EXPECT_EQ(g.get_op(1).type, OpType::Add);
    EXPECT_EQ(g.get_op(2).type, OpType::Softmax);
    EXPECT_EQ(g.get_op(3).type, OpType::LayerNorm);
    EXPECT_EQ(g.get_op(4).type, OpType::GELU);
    EXPECT_EQ(g.get_op(5).type, OpType::ReLU);
}

TEST(ParserTest, InvalidOpType) {
    std::string ir = R"(
Op 0: invalid_op
  shape: [32]
)";

    EXPECT_THROW(Parser::parse(ir), std::invalid_argument);
}

TEST(ParserTest, MissingShape) {
    std::string ir = R"(
Op 0: input
  no_shape_here: [32]
)";

    // Should parse but shape will be empty
    Graph g = Parser::parse(ir);
    const auto& op = g.get_op(0);
    EXPECT_EQ(op.output_shape.dims.size(), 0);
}

// ============================================================================
// Printer Tests
// ============================================================================

TEST(PrinterTest, BasicPrint) {
    Graph g;
    Op op0(0, OpType::Input, {}, Shape({16, 16}));
    Op op1(1, OpType::Add, {0, 0}, Shape({16, 16}));

    g.add_op(op0);
    g.add_op(op1);

    std::string output = Printer::print(g);
    EXPECT_NE(output.find("Op 0: input"), std::string::npos);
    EXPECT_NE(output.find("Op 1: add"), std::string::npos);
    EXPECT_NE(output.find("[16,16]"), std::string::npos);
}

TEST(PrinterTest, WithAttributes) {
    Graph g;
    Op op(0, OpType::MatMul, {}, Shape({128, 128}));
    op.add_attribute("tile_m", "16");
    op.add_attribute("tile_n", "16");

    g.add_op(op);

    std::string output = Printer::print(g);
    EXPECT_NE(output.find("tile_m=16"), std::string::npos);
    EXPECT_NE(output.find("tile_n=16"), std::string::npos);
}

// ============================================================================
// Validator Tests
// ============================================================================

TEST(ValidatorTest, ShapeBroadcasting) {
    Shape s1({3, 4, 5});
    Shape s2({1, 5});
    EXPECT_TRUE(Validator::shapes_broadcastable(s1, s2));

    Shape s3({3, 4});
    Shape s4({1, 4});
    EXPECT_TRUE(Validator::shapes_broadcastable(s3, s4));

    Shape s5({3, 4});
    Shape s6({3, 5});
    EXPECT_FALSE(Validator::shapes_broadcastable(s5, s6));
}

TEST(ValidatorTest, BroadcastShape) {
    Shape s1({1, 5});
    Shape s2({3, 1});
    Shape result = Validator::broadcast_shape(s1, s2);
    EXPECT_EQ(result.str(), "[3,5]");
}

TEST(ValidatorTest, BroadcastShapeIncompatible) {
    Shape s1({3, 4});
    Shape s2({3, 5});
    EXPECT_THROW(Validator::broadcast_shape(s1, s2), std::invalid_argument);
}

TEST(ValidatorTest, ValidateSimpleGraph) {
    Graph g;
    Op op0(0, OpType::Input, {}, Shape({16, 16}));
    Op op1(1, OpType::Add, {0, 0}, Shape({16, 16}));

    g.add_op(op0);
    g.add_op(op1);

    auto errors = Validator::validate(g);
    EXPECT_EQ(errors.size(), 0);
}

TEST(ValidatorTest, ValidateMatMulShapeMismatch) {
    Graph g;
    Op op0(0, OpType::Input, {}, Shape({16, 16}));
    Op op1(1, OpType::Input, {}, Shape({32, 32}));  // K dimension mismatch
    Op op2(2, OpType::MatMul, {0, 1}, Shape({16, 32}));

    g.add_op(op0);
    g.add_op(op1);
    g.add_op(op2);

    auto errors = Validator::validate(g);
    EXPECT_GT(errors.size(), 0);
    EXPECT_NE(errors[0].find("inner dims mismatch"), std::string::npos);
}

TEST(ValidatorTest, ValidateMatMulInvalidDims) {
    Graph g;
    Op op0(0, OpType::Input, {}, Shape({16}));  // Not a matrix
    Op op1(1, OpType::Input, {}, Shape({16, 16}));
    Op op2(2, OpType::MatMul, {0, 1}, Shape({16, 16}));

    g.add_op(op0);
    g.add_op(op1);
    g.add_op(op2);

    auto errors = Validator::validate(g);
    EXPECT_GT(errors.size(), 0);
    EXPECT_NE(errors[0].find("not matrix"), std::string::npos);
}

// ============================================================================
// Round-trip Tests
// ============================================================================

TEST(RoundTripTest, ParsePrintParse) {
    std::string original = R"(
Op 0: input
  shape: [16,16]

Op 1: add
  inputs: [0,0]
  shape: [16,16]

Op 2: gelu
  inputs: [1]
  shape: [16,16]
)";

    Graph g1 = Parser::parse(original);
    std::string printed = Printer::print(g1);
    Graph g2 = Parser::parse(printed);

    EXPECT_EQ(g1.num_ops(), g2.num_ops());
    for (auto op_id : g1.get_op_ids()) {
        const auto& op1 = g1.get_op(op_id);
        const auto& op2 = g2.get_op(op_id);
        EXPECT_EQ(op1.type, op2.type);
        EXPECT_EQ(op1.output_shape.str(), op2.output_shape.str());
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
