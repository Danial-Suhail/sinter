#include "sinter/passes.h"

#include <unordered_map>
#include <unordered_set>

namespace sinter {

    Graph FusionPass::optimize(const Graph& graph) {
        Graph result;
        const auto topo_order = graph.topological_sort();

        std::unordered_map<size_t, size_t> consumer_count;
        for (size_t op_id : topo_order) {
            consumer_count[op_id] = 0;
        }
        for (size_t op_id : topo_order) {
            const auto& op = graph.get_op(op_id);
            for (size_t input_id : op.input_ids) {
                if (consumer_count.find(input_id) == consumer_count.end()) {
                    consumer_count[input_id] = 0;
                }
                consumer_count[input_id]++;
            }
        }

        std::unordered_map<size_t, size_t> relu_for_add;
        std::unordered_set<size_t> fused_relu_ids;

        for (size_t op_id : topo_order) {
            const auto& relu = graph.get_op(op_id);
            if (relu.type != OpType::ReLU || relu.input_ids.size() != 1) {
                continue;
            }

            const size_t add_id = relu.input_ids[0];
            if (consumer_count[add_id] != 1) {
                continue;
            }

            const auto& add = graph.get_op(add_id);
            if (!can_fuse(add, relu)) {
                continue;
            }

            relu_for_add[add_id] = op_id;
            fused_relu_ids.insert(op_id);
        }

        std::unordered_map<size_t, size_t> replacement;
        for (size_t op_id : topo_order) {
            replacement[op_id] = op_id;
        }
        for (const auto& [add_id, relu_id] : relu_for_add) {
            replacement[relu_id] = add_id;
        }

        auto resolve_id = [&replacement](size_t op_id) {
            while (replacement.find(op_id) != replacement.end() && replacement[op_id] != op_id) {
                op_id = replacement[op_id];
            }
            return op_id;
        };

        for (size_t op_id : topo_order) {
            if (fused_relu_ids.find(op_id) != fused_relu_ids.end()) {
                continue;
            }

            const auto& op = graph.get_op(op_id);
            Op rewritten = op;

            rewritten.input_ids.clear();
            rewritten.input_ids.reserve(op.input_ids.size());
            for (size_t input_id : op.input_ids) {
                rewritten.input_ids.push_back(resolve_id(input_id));
            }

            if (relu_for_add.find(op_id) != relu_for_add.end()) {
                const auto& relu = graph.get_op(relu_for_add.at(op_id));
                rewritten.type = OpType::FusedOp;
                rewritten.output_shape = relu.output_shape;
                rewritten.attributes.clear();
                rewritten.add_attribute("fused_ops", "add,relu");
            }

            result.add_op(rewritten);
        }

        return result;
    }

    bool FusionPass::can_fuse(const Op& a, const Op& b) const {
        if (a.type != OpType::Add || b.type != OpType::ReLU) {
            return false;
        }

        if (b.input_ids.size() != 1 || b.input_ids[0] != a.id) {
            return false;
        }

        if (a.output_shape != b.output_shape) {
            return false;
        }

        return true;
    }

    Graph TilingPass::optimize(const Graph& graph) {
        // TODO: Implement tiling logic
        return graph;
    }

    void TilingPass::annotate_tile_config(Op& op, const Shape& output_shape) {
        (void)op;  // Suppress unused parameter warning
        (void)output_shape;
        // TODO: Implement tile size selection
    }

    Graph LayoutPass::optimize(const Graph& graph) {
        // TODO: Implement layout transformation
        return graph;
    }

    bool LayoutPass::needs_transpose(const Op& op, const Op& consumer) const {
        (void)op;  // Suppress unused parameter warning
        (void)consumer;
        // TODO: Implement transpose detection
        return false;
    }

    Graph PassManager::run(const Graph& graph) {
        Graph current = graph;
        for (auto& pass : passes) {
            current = pass->optimize(current);
        }
        return current;
    }

} 
