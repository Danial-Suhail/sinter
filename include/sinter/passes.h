/**
 * @file passes.h
 * @brief Optimization passes (fusion, tiling, memory layout)
 */

#pragma once

#include "sinter/ir.h"

namespace sinter {

/**
 * @class Pass
 * @brief Abstract base class for IR optimization passes
 */
class Pass {
public:
    virtual ~Pass() = default;

    /**
     * Apply optimization pass to graph
     * @param graph Input graph (may be modified)
     * @return Optimized graph
     */
    virtual Graph optimize(const Graph& graph) = 0;
};

/**
 * @class FusionPass
 * @brief Merge compatible operations into fused kernels
 * Example: add + layernorm -> fused_layernorm
 */
class FusionPass : public Pass {
public:
    Graph optimize(const Graph& graph) override;

private:
    bool can_fuse(const Op& a, const Op& b) const;
};

/**
 * @class TilingPass
 * @brief Annotate operations with tile sizes for better cache usage
 */
class TilingPass : public Pass {
public:
    Graph optimize(const Graph& graph) override;

private:
    void annotate_tile_config(Op& op, const Shape& output_shape);
};

/**
 * @class LayoutPass
 * @brief Transform tensor memory layout (row-major, col-major, etc.)
 */
class LayoutPass : public Pass {
public:
    Graph optimize(const Graph& graph) override;

private:
    bool needs_transpose(const Op& op, const Op& consumer) const;
};

/**
 * @class PassManager
 * @brief Orchestrate multiple passes
 */
class PassManager {
public:
    void add_pass(std::unique_ptr<Pass> pass) {
        passes.push_back(std::move(pass));
    }

    /**
     * Apply all passes in order
     * @param graph Input graph
     * @return Optimized graph
     */
    Graph run(const Graph& graph);

private:
    std::vector<std::unique_ptr<Pass>> passes;
};

} // namespace sinter
