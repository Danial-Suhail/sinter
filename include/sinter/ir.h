/**
 * @file ir.h
 * @brief Tensor Compiler IR (Intermediate Representation) definitions
 *
 * This file defines the core IR for the Sinter tensor compiler:
 * - OpType: enum of supported tensor operations
 * - Op: individual operation node
 * - Graph: DAG of operations with shape validation
 * - Parser: parse text IR format
 * - Printer: pretty-print IR for debugging
 * - Validator: check shape consistency and topology
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <sstream>

namespace sinter {

    /**
     * @enum OpType
     * @brief Supported tensor operations
     */
    enum class OpType {
        MatMul,          // Matrix multiplication
        Add,             // Element-wise addition
        Softmax,         // Softmax normalization
        LayerNorm,       // Layer normalization
        GELU,            // GELU activation
        FlashAttention,  // Flash attention kernel
        ReLU,            // ReLU activation
        Multiply,        // Element-wise multiplication
        Reshape,         // Tensor reshape
        Transpose,       // Tensor transpose
        FusedOp,         // Fused multi-op kernel
        Input,           // Graph input (no computation)
        Output,          // Graph output (no computation)
    };

    /**
     * @struct Shape
     * @brief Tensor shape representation
     */
    struct Shape {
        std::vector<size_t> dims;

        Shape() = default;
        explicit Shape(const std::vector<size_t>& d) : dims(d) {}
        
        size_t numel() const {
            size_t n = 1;
            for (size_t d : dims) n *= d;
            return n;
        }

        bool operator==(const Shape& other) const {
            return dims == other.dims;
        }

        bool operator!=(const Shape& other) const {
            return !(*this == other);
        }

        std::string str() const {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < dims.size(); ++i) {
                if (i > 0) oss << ",";
                oss << dims[i];
            }
            oss << "]";
            return oss.str();
        }
    };

    /**
     * @struct OpAttribute
     * @brief Operation-specific attributes (tile shapes, strides, etc.)
     */
    struct OpAttribute {
        std::string key;
        std::string value;

        OpAttribute() = default;
        OpAttribute(const std::string& k, const std::string& v) : key(k), value(v) {}
    };

    /**
     * @struct Op
     * @brief Single operation node in the IR graph
     */
    struct Op {
        size_t id;
        OpType type;
        std::vector<size_t> input_ids;  // IDs of input ops
        size_t output_id;               // Unique ID for this op's output
        Shape output_shape;
        std::vector<OpAttribute> attributes;

        Op() = default;
        Op(size_t id, OpType type, const std::vector<size_t>& inputs, const Shape& shape)
            : id(id), type(type), input_ids(inputs), output_id(id), output_shape(shape) {}

        void add_attribute(const std::string& key, const std::string& value) {
            attributes.emplace_back(key, value);
        }

        std::string get_attribute(const std::string& key) const {
            for (const auto& attr : attributes) {
                if (attr.key == key) return attr.value;
            }
            return "";
        }

        std::string op_type_str() const {
            switch (type) {
                case OpType::MatMul: return "matmul";
                case OpType::Add: return "add";
                case OpType::Softmax: return "softmax";
                case OpType::LayerNorm: return "layernorm";
                case OpType::GELU: return "gelu";
                case OpType::FlashAttention: return "flash_attention";
                case OpType::ReLU: return "relu";
                case OpType::Multiply: return "multiply";
                case OpType::Reshape: return "reshape";
                case OpType::Transpose: return "transpose";
                case OpType::FusedOp: return "fused_op";
                case OpType::Input: return "input";
                case OpType::Output: return "output";
                default: return "unknown";
            }
        }
    };

    /**
     * @class Graph
     * @brief Computational graph (DAG) of tensor operations
     */
    class Graph {
    public:
        Graph() = default;

        /**
         * Add an operation to the graph
         * @param op Operation to add
         * @throws std::invalid_argument if op is invalid
         */
        void add_op(const Op& op) {
            ops[op.id] = op;
            input_shapes[op.output_id] = op.output_shape;
        }

        /**
         * Get operation by ID
         * @param op_id Operation ID
         * @return Reference to operation
         * @throws std::out_of_range if op_id not found
         */
        Op& get_op(size_t op_id) {
            return ops.at(op_id);
        }

        const Op& get_op(size_t op_id) const {
            return ops.at(op_id);
        }

        /**
         * Get all operations in topological order
         * @return Vector of operation IDs in execution order
         */
        std::vector<size_t> topological_sort() const;

        /**
         * Get output shape of operation
         * @param op_id Operation ID
         * @return Output shape
         */
        Shape get_output_shape(size_t op_id) const {
            return ops.at(op_id).output_shape;
        }

        /**
         * Check if graph is valid (no cycles, shape consistency)
         * @return true if valid
         * @throws std::invalid_argument with error message if invalid
         */
        bool validate();

        /**
         * Get all operation IDs
         * @return Vector of operation IDs
         */
        std::vector<size_t> get_op_ids() const {
            std::vector<size_t> ids;
            for (const auto& p : ops) {
                ids.push_back(p.first);
            }
            return ids;
        }

        size_t num_ops() const {
            return ops.size();
        }

        void clear() {
            ops.clear();
            input_shapes.clear();
        }

    private:
        std::unordered_map<size_t, Op> ops;
        std::unordered_map<size_t, Shape> input_shapes;
    };

    /**
     * @class Parser
     * @brief Parse text-based IR format into Graph
     *
     * Text IR format example:
     *   Op 0: input
     *     shape: [16,16,16]
     *
     *   Op 1: matmul
     *     inputs: [0, 1]
     *     output_shape: [16,16]
     *
     *   Op 2: add
     *     inputs: [1, bias_tensor_id]
     *     output_shape: [16,16]
     */
    class Parser {
    public:
        /**
         * Parse IR from string
         * @param ir_text IR text
         * @return Parsed graph
         * @throws std::invalid_argument if parsing fails
         */
        static Graph parse(const std::string& ir_text);

        /**
         * Parse IR from file
         * @param filename Path to IR file
         * @return Parsed graph
         * @throws std::runtime_error if file cannot be read
         */
        static Graph parse_file(const std::string& filename);

    private:
        static OpType str_to_optype(const std::string& s);
        static std::vector<std::string> split_line(const std::string& line);
    };

    /**
     * @class Printer
     * @brief Pretty-print IR for debugging
     */
    class Printer {
    public:
        /**
         * Convert graph to human-readable string
         * @param graph Graph to print
         * @return Formatted IR string
         */
        static std::string print(const Graph& graph);

        /**
         * Write graph to file
         * @param graph Graph to write
         * @param filename Output filename
         * @throws std::runtime_error if file cannot be written
         */
        static void print_file(const Graph& graph, const std::string& filename);
    };

    /**
     * @class Validator
     * @brief Validate IR correctness
     */
    class Validator {
    public:
        /**
         * Validate graph structure and shapes
         * @param graph Graph to validate
         * @return Validation errors (empty if valid)
         */
        static std::vector<std::string> validate(const Graph& graph);

        /**
         * Check for shape consistency on binary ops
         * @param a Shape of first input
         * @param b Shape of second input
         * @return true if shapes are broadcastable
         */
        static bool shapes_broadcastable(const Shape& a, const Shape& b);

        /**
         * Compute broadcast result shape
         * @param a Shape of first input
         * @param b Shape of second input
         * @return Result shape
         * @throws std::invalid_argument if shapes not broadcastable
         */
        static Shape broadcast_shape(const Shape& a, const Shape& b);
    };
    
}
