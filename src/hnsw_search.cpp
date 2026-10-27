#include "hnsw.hpp"
#include "exact_search.hpp"

#include <algorithm>
#include <utility>
#include <stdexcept>
#include <cmath> 
#include <random>

int HNSW:: greedy_search(vector <float> query, int entry_point,int level){

    if (entry_point < 0 || entry_point >= size())
        throw out_of_range("Invalid entry point");

    if (level < 0 || level > nodes[entry_point].level)
        throw out_of_range("Invalid layer");


    int current = entry_point;

    //Node curr_node = get_Node(current);
    float distance = squared_distance(query,nodes[current].value);

    bool i= true;
    while(i){
        i = false;
        for(auto neighbor_id: nodes[current].neighbour[level]){
            float d = squared_distance(query,nodes[neighbor_id].value);
            if(distance>d){
                distance = d;
                i = true;
                current = neighbor_id;
                break;
            }
        }
    }
    return current;
}

vector<int> HNSW::search_layer(vector<float>& query,int entry_point,int ef,int layer) {
    if (ef <= 0) {
        throw std::invalid_argument("ef must be positive");
    }

    if (entry_point < 0 || entry_point >= size()) {
        throw std::out_of_range("Invalid entry point");
    }

    if (layer < 0 || layer > get_Node(entry_point).level) {
        throw out_of_range("Entry point does not exist on this layer");
    }

    using Item = pair<float, int>;

    // C: min-heap; closest candidate first.
    priority_queue<Item,vector<Item>,greater<Item>> candidates;

    // W: max-heap; farthest retained candidate first.
    priority_queue<Item> best;

    // V: visited nodes.
    vector<bool> visited(size(), false);

    float initial_dist = squared_distance(query, get_Node(entry_point).value);

    candidates.push({initial_dist, entry_point});
    best.push({initial_dist, entry_point});
    visited[entry_point] = true;

    while (!candidates.empty()) {
        Item current = candidates.top();

        // Stop if the closest unexplored candidate is
        // farther than the worst retained candidate.
        if (best.size() >=  static_cast<size_t>(ef) && current.first > best.top().first) {
            break;
        }

        candidates.pop();

        int current_id = current.second;

        for (int neighbor_id : get_Node(current_id).neighbour[layer]) {

            if (visited[neighbor_id]) {
                continue;
            }

            visited[neighbor_id] = true;

            float neighbor_dist = squared_distance(query, get_Node(neighbor_id).value);

            bool promising = best.size() <  static_cast<size_t>(ef) || neighbor_dist < best.top().first;

            if (promising) { 
                candidates.push({neighbor_dist, neighbor_id});
                best.push({neighbor_dist, neighbor_id});

                if (best.size() >  static_cast<size_t>(ef)) {
                    best.pop();
                }
            }
        }
    }

    // Convert retained candidates to ascending distance order.
    vector<Item> ordered;

    while (!best.empty()) {
        ordered.push_back(best.top());
        best.pop();
    }

    sort(ordered.begin(), ordered.end());

    vector<int> result;
    result.reserve(ordered.size());

    for (auto& item : ordered) {
        result.push_back(item.second);
    }

    return result;
}


vector <int> HNSW::knn_search(vector<float>& query,int k,int ef_search) {
    if (nodes.empty()) {
        throw invalid_argument("Cannot search an empty index");
    }

    if (k < 1 || k > size()) {
        throw invalid_argument("Invalid k");
    }

    if (ef_search < 1) {
        throw invalid_argument("ef_search must be positive");
    }

    if (query.size() != nodes[0].value.size()) {
        throw invalid_argument("Dimension mismatch");
    }

    // ef must be at least k to return k results.
    int ef = max(k, ef_search);

    int current_entry = entry_point;

    // Greedily descend from the highest layer to layer 1.
    for (int layer = max_level; layer > 0; --layer) {
        current_entry = greedy_search(query, current_entry, layer);
    }

    // Search the base layer for the final candidates.
    vector<int> candidates =
        search_layer(query, current_entry, ef, 0);

    // search_layer returns candidates in increasing distance order.
    if (static_cast<int>(candidates.size()) > k) {
        candidates.resize(k);
    }

    return candidates;
}


