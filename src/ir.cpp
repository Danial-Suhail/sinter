/**
 * @file ir.cpp
 * @brief Implementation of tensor compiler IR
 */

#include "sinter/ir.h"
#include <fstream>
#include <algorithm>
#include <sstream>
#include <cctype>

namespace sinter {

    // ============================================================================
    // Graph Implementation
    // ============================================================================

    std::vector<size_t> Graph::topological_sort() const {
        std::vector<size_t> result;
        std::unordered_map<size_t, int> in_degree;
        std::unordered_map<size_t, std::vector<size_t>> adj;

        // Initialize in-degree and adjacency list
        for (const auto& [op_id, op] : ops) {
            if (in_degree.find(op_id) == in_degree.end()) {
                in_degree[op_id] = 0;
            }
            for (size_t input_id : op.input_ids) {
                adj[input_id].push_back(op_id);
                in_degree[op_id]++;
            }
        }

        // Kahn's algorithm
        std::vector<size_t> queue;
        for (const auto& [op_id, degree] : in_degree) {
            if (degree == 0) {
                queue.push_back(op_id);
            }
        }

        while (!queue.empty()) {
            size_t u = queue.back();
            queue.pop_back();
            result.push_back(u);

            for (size_t v : adj[u]) {
                in_degree[v]--;
                if (in_degree[v] == 0) {
                    queue.push_back(v);
                }
            }
        }

        if (result.size() != ops.size()) {
            throw std::invalid_argument("Graph contains cycle");
        }

        return result;
    }

    bool Graph::validate() {
        // Check for cycles
        try {
            topological_sort();
        } catch (const std::invalid_argument&) {
            return false;
        }

        // Check shape consistency
        for (const auto& [op_id, op] : ops) {
            switch (op.type) {
                case OpType::MatMul: {
                    if (op.input_ids.size() < 2) return false;
                    const auto& a = get_output_shape(op.input_ids[0]);
                    const auto& b = get_output_shape(op.input_ids[1]);
                    if (a.dims.size() < 2 || b.dims.size() < 2) return false;
                    if (a.dims[a.dims.size() - 1] != b.dims[b.dims.size() - 2]) return false;
                    break;
                }
                case OpType::Add:
                case OpType::Multiply: {
                    if (op.input_ids.size() < 2) return false;
                    const auto& a = get_output_shape(op.input_ids[0]);
                    const auto& b = get_output_shape(op.input_ids[1]);
                    if (!Validator::shapes_broadcastable(a, b)) return false;
                    break;
                }
                case OpType::Softmax:
                case OpType::LayerNorm:
                case OpType::GELU:
                case OpType::ReLU: {
                    if (op.input_ids.empty()) return false;
                    break;
                }
                default:
                    break;
            }
        }

        return true;
    }

    // ============================================================================
    // Parser Implementation
    // ============================================================================

    OpType Parser::str_to_optype(const std::string& s) {
        if (s == "matmul") return OpType::MatMul;
        if (s == "add") return OpType::Add;
        if (s == "softmax") return OpType::Softmax;
        if (s == "layernorm") return OpType::LayerNorm;
        if (s == "gelu") return OpType::GELU;
        if (s == "flash_attention") return OpType::FlashAttention;
        if (s == "relu") return OpType::ReLU;
        if (s == "multiply") return OpType::Multiply;
        if (s == "reshape") return OpType::Reshape;
        if (s == "transpose") return OpType::Transpose;
        if (s == "fused_op") return OpType::FusedOp;
        if (s == "input") return OpType::Input;
        if (s == "output") return OpType::Output;
        throw std::invalid_argument("Unknown op type: " + s);
    }

    std::vector<std::string> Parser::split_line(const std::string& line) {
        std::vector<std::string> tokens;
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) {
            // Remove trailing punctuation
            while (!token.empty() && (token.back() == ',' || token.back() == ':')) {
                token.pop_back();
            }
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }
        return tokens;
    }

    Graph Parser::parse(const std::string& ir_text) {
        Graph graph;
        std::istringstream iss(ir_text);
        std::string line;

        bool has_current_op = false;
        size_t current_op_id = 0;
        OpType current_type = OpType::Input;
        std::vector<size_t> current_inputs;
        Shape current_shape;
        std::vector<OpAttribute> current_attrs;

        auto flush_op = [&]() {
            if (!has_current_op) {
                return;
            }

            {
                Op op(current_op_id, current_type, current_inputs, current_shape);
                for (const auto& attr : current_attrs) {
                    op.add_attribute(attr.key, attr.value);
                }
                graph.add_op(op);
                current_inputs.clear();
                current_attrs.clear();
            }
            has_current_op = false;
        };

        while (std::getline(iss, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            if (line.empty() || line[0] == '#') continue;

            // Parse "Op <id>: <type>"
            if (line.find("Op ") == 0) {
                flush_op();
                auto tokens = split_line(line);
                if (tokens.size() < 3) throw std::invalid_argument("Invalid op line: " + line);
                current_op_id = std::stoul(tokens[1]);
                current_type = str_to_optype(tokens[2]);
                current_shape = Shape{};
                has_current_op = true;
                continue;
            }

            // Parse "shape: [d1,d2,...]"
            if (line.find("shape:") != std::string::npos) {
                size_t start = line.find('[');
                size_t end = line.find(']');
                if (start == std::string::npos || end == std::string::npos) {
                    throw std::invalid_argument("Invalid shape line: " + line);
                }
                std::string shape_str = line.substr(start + 1, end - start - 1);
                std::vector<size_t> dims;
                std::istringstream shape_iss(shape_str);
                std::string dim_str;
                while (std::getline(shape_iss, dim_str, ',')) {
                    dim_str.erase(0, dim_str.find_first_not_of(' '));
                    dim_str.erase(dim_str.find_last_not_of(' ') + 1);
                    if (!dim_str.empty()) {
                        dims.push_back(std::stoul(dim_str));
                    }
                }
                current_shape = Shape(dims);
                continue;
            }

            // Parse "inputs: [id1,id2,...]"
            if (line.find("inputs:") != std::string::npos) {
                size_t start = line.find('[');
                size_t end = line.find(']');
                if (start == std::string::npos || end == std::string::npos) {
                    throw std::invalid_argument("Invalid inputs line: " + line);
                }
                std::string inputs_str = line.substr(start + 1, end - start - 1);
                std::istringstream inputs_iss(inputs_str);
                std::string id_str;
                while (std::getline(inputs_iss, id_str, ',')) {
                    id_str.erase(0, id_str.find_first_not_of(' '));
                    id_str.erase(id_str.find_last_not_of(' ') + 1);
                    if (!id_str.empty()) {
                        current_inputs.push_back(std::stoul(id_str));
                    }
                }
                continue;
            }

            // Parse "attr: key=value"
            if (line.find("attr:") != std::string::npos) {
                size_t eq_pos = line.find('=');
                if (eq_pos == std::string::npos) continue;
                std::string key = line.substr(line.find(':') + 1, eq_pos - line.find(':') - 1);
                std::string value = line.substr(eq_pos + 1);
                key.erase(0, key.find_first_not_of(' '));
                key.erase(key.find_last_not_of(' ') + 1);
                value.erase(0, value.find_first_not_of(' '));
                value.erase(value.find_last_not_of(' ') + 1);
                current_attrs.emplace_back(key, value);
                continue;
            }
        }

        flush_op();
        return graph;
    }

    Graph Parser::parse_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return parse(buffer.str());
    }

    // ============================================================================
    // Printer Implementation
    // ============================================================================

    std::string Printer::print(const Graph& graph) {
        std::ostringstream oss;
        auto op_ids = graph.get_op_ids();
        std::sort(op_ids.begin(), op_ids.end());

        for (size_t op_id : op_ids) {
            const auto& op = graph.get_op(op_id);
            oss << "Op " << op.id << ": " << op.op_type_str() << "\n";
            oss << "  output_shape: " << op.output_shape.str() << "\n";

            if (!op.input_ids.empty()) {
                oss << "  inputs: [";
                for (size_t i = 0; i < op.input_ids.size(); ++i) {
                    if (i > 0) oss << ", ";
                    oss << op.input_ids[i];
                }
                oss << "]\n";
            }

            for (const auto& attr : op.attributes) {
                oss << "  attr: " << attr.key << "=" << attr.value << "\n";
            }
            oss << "\n";
        }

        return oss.str();
    }

    void Printer::print_file(const Graph& graph, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
        file << print(graph);
    }

    // ============================================================================
    // Validator Implementation
    // ============================================================================

    std::vector<std::string> Validator::validate(const Graph& graph) {
        std::vector<std::string> errors;

        // Check for cycles
        try {
            graph.topological_sort();
        } catch (const std::invalid_argument& e) {
            errors.push_back("Cycle detected: " + std::string(e.what()));
            return errors;
        }

        // Check shape consistency for each op
        for (size_t op_id : graph.get_op_ids()) {
            const auto& op = graph.get_op(op_id);

            switch (op.type) {
                case OpType::MatMul: {
                    if (op.input_ids.size() < 2) {
                        errors.push_back("MatMul op " + std::to_string(op.id) + " requires 2 inputs");
                        continue;
                    }
                    const auto& a = graph.get_output_shape(op.input_ids[0]);
                    const auto& b = graph.get_output_shape(op.input_ids[1]);
                    if (a.dims.size() < 2) {
                        errors.push_back("MatMul op " + std::to_string(op.id) + " input 0 is not matrix");
                    }
                    if (b.dims.size() < 2) {
                        errors.push_back("MatMul op " + std::to_string(op.id) + " input 1 is not matrix");
                    }
                    if (a.dims[a.dims.size() - 1] != b.dims[b.dims.size() - 2]) {
                        errors.push_back("MatMul op " + std::to_string(op.id) + " inner dims mismatch: " +
                                    std::to_string(a.dims[a.dims.size() - 1]) + " vs " +
                                    std::to_string(b.dims[b.dims.size() - 2]));
                    }
                    break;
                }
                case OpType::Add:
                case OpType::Multiply: {
                    if (op.input_ids.size() < 2) {
                        errors.push_back(op.op_type_str() + " op " + std::to_string(op.id) + " requires 2 inputs");
                        continue;
                    }
                    const auto& a = graph.get_output_shape(op.input_ids[0]);
                    const auto& b = graph.get_output_shape(op.input_ids[1]);
                    if (!shapes_broadcastable(a, b)) {
                        errors.push_back(op.op_type_str() + " op " + std::to_string(op.id) +
                                    " inputs not broadcastable: " + a.str() + " vs " + b.str());
                    }
                    break;
                }
                default:
                    break;
            }
        }

        return errors;
    }

    bool Validator::shapes_broadcastable(const Shape& a, const Shape& b) {
        size_t a_ndim = a.dims.size();
        size_t b_ndim = b.dims.size();
        size_t max_ndim = std::max(a_ndim, b_ndim);

        for (size_t i = 0; i < max_ndim; ++i) {
            size_t a_dim = (i < a_ndim) ? a.dims[a_ndim - 1 - i] : 1;
            size_t b_dim = (i < b_ndim) ? b.dims[b_ndim - 1 - i] : 1;

            if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                return false;
            }
        }

        return true;
    }

    Shape Validator::broadcast_shape(const Shape& a, const Shape& b) {
        if (!shapes_broadcastable(a, b)) {
            throw std::invalid_argument("Shapes not broadcastable: " + a.str() + " vs " + b.str());
        }

        size_t a_ndim = a.dims.size();
        size_t b_ndim = b.dims.size();
        size_t max_ndim = std::max(a_ndim, b_ndim);
        std::vector<size_t> result_dims(max_ndim);

        for (size_t i = 0; i < max_ndim; ++i) {
            size_t a_dim = (i < a_ndim) ? a.dims[a_ndim - 1 - i] : 1;
            size_t b_dim = (i < b_ndim) ? b.dims[b_ndim - 1 - i] : 1;
            result_dims[max_ndim - 1 - i] = std::max(a_dim, b_dim);
        }

        return Shape(result_dims);
    }

}
