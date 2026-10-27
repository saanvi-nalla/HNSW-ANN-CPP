#include "hnsw.hpp"
#include "exact_search.hpp"

#include <algorithm>
#include <utility>
#include <stdexcept>
#include <cmath> 
#include <random>

int HNSW::insert(vector<float>& value) {
    // Validate dimensions before changing the graph.
    if (value.empty()) {
        throw invalid_argument("Vector cannot be empty");
    }

    if (!nodes.empty() && value.size() != nodes[0].value.size()) {
        throw invalid_argument("Dimension mismatch");
    }

    int level = random_level();

    // Special case: inserting the first node.
    if (nodes.empty()) {
        int id = add_node(value, level);
        entry_point = id;
        max_level = level;
        return id;
    }

    int new_id = add_node(value, level);
    int current_entry = entry_point;

    // Descend through layers above the new node's level.
    for (int layer = max_level; layer > level; --layer) {
        current_entry = greedy_search(value, current_entry, layer);
    }

    // Connect the new node on every layer it belongs to.
    for (int layer = min(level, max_level); layer >= 0; --layer) {
        vector<int> candidates =
            search_layer(value, current_entry, ef_construction, layer);

        int max_degree = (layer == 0) ? 2 * M : M;

        // search_layer returns candidates in increasing distance order.
        int limit = min(max_degree, static_cast<int>(candidates.size()));

        for (int i = 0; i < limit; ++i) {
            int neighbor_id = candidates[i];
            add_edge(new_id, neighbor_id, layer);
        }

        // Prune the new node and its neighbors if their degrees exceed
        // the layer's capacity. Pruning also removes reciprocal edges.
        prune_neighbours(new_id, layer, max_degree);

        for (int i = 0; i < limit; ++i) {
            int neighbor_id = candidates[i];
            prune_neighbours(neighbor_id, layer, max_degree);
        }

        // Use the closest candidate as the entry for the next layer.
        if (!candidates.empty()) {
            current_entry = candidates[0];
        }
    }

    // A node reaching a new highest layer becomes the entry point.
    if (level > max_level) {
        entry_point = new_id;
        max_level = level;
    }

    return new_id;
}