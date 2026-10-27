#include "hnsw.hpp"
#include "exact_search.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

void HNSW::prune_neighbours(int node_id, int layer, int max_degree) {

    if (node_id < 0 || node_id >= size())
    throw out_of_range("Invalid node index");

    if (layer < 0 || layer > nodes[node_id].level)
    throw out_of_range("Invalid layer");

    if (max_degree < 1)
    throw invalid_argument("max_degree must be positive");

    auto& neighbors = nodes[node_id].neighbour[layer];

    if (neighbors.size() <=  static_cast<size_t>(max_degree))
        return;

    std::sort(neighbors.begin(), neighbors.end(),
        [&](int a, int b) {
            return squared_distance(nodes[node_id].value,
                                    nodes[a].value)
                 < squared_distance(nodes[node_id].value,
                                    nodes[b].value);
        });

    vector<int> removed(neighbors.begin() + max_degree, neighbors.end());

    neighbors.resize(max_degree);

    for (int other : removed) {
        auto& reverse = nodes[other].neighbour[layer];

        reverse.erase(
            std::remove(reverse.begin(), reverse.end(), node_id),
            reverse.end());
    }
}

std::vector<int> HNSW::search_neighbor(const std::vector<float>& query,const std::vector<int>& candidates,int max_degree) {
    if (max_degree < 0) {
        throw std::invalid_argument(
            "max_degree must be non-negative"
        );
    }

    if (max_degree == 0 || candidates.empty()) {
        return {};
    }

    std::vector<int> sorted_candidates = candidates;

    // Sort candidates by distance from the query.
    std::sort(
        sorted_candidates.begin(),
        sorted_candidates.end(),
        [&](int a, int b) {
            return squared_distance(query, nodes[a].value) <
                   squared_distance(query, nodes[b].value);
        }
    );

    std::vector<int> selected;
    selected.reserve(
        std::min(max_degree,
                 static_cast<int>(sorted_candidates.size()))
    );

    for (int candidate : sorted_candidates) {
        bool good = true;

        // Check whether this candidate is redundant.
        for (int chosen : selected) {
            double candidate_to_chosen = squared_distance(
                nodes[candidate].value,
                nodes[chosen].value
            );

            double candidate_to_query = squared_distance(
                query,
                nodes[candidate].value
            );

            if (candidate_to_chosen < candidate_to_query) {
                good = false;
                break;
            }
        }

        if (good) {
            selected.push_back(candidate);
        }

        if (static_cast<int>(selected.size()) >= max_degree) {
            break;
        }
    }

    return selected;
}