#include "hnsw.hpp"
#include "exact_search.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

template <typename Exception, typename Function>
bool throws_as(Function function) {
    try {
        function();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

static bool contains(const vector<int>& values, int value) {
    return find(values.begin(), values.end(), value) != values.end();
}

#define CHECK(condition) do { \
    if (!(condition)) { \
        cerr << "FAILED at line " << __LINE__ << ": " #condition << '\n'; \
        return 1; \
    } \
    ++checks; \
} while (false)

int main() {
    int checks = 0;

    // Constructor and initial state
    CHECK(throws_as<invalid_argument>([] { HNSW bad(1, 10); }));
    CHECK(throws_as<invalid_argument>([] { HNSW bad(4, 0); }));
    HNSW empty(4, 20);
    CHECK(empty.size() == 0);
    CHECK(empty.get_entry_point() == -1);
    CHECK(empty.get_max_level() == -1);

    // add_node, get_Node, size
    vector<float> a{0, 0}, b{1, 0};
    CHECK(throws_as<invalid_argument>([&] {
        vector<float> v{1, 0};
        empty.add_node(v, -1);
    }));
    int a_id = empty.add_node(a, 0);
    int b_id = empty.add_node(b, 1);
    CHECK(a_id == 0 && b_id == 1);
    CHECK(empty.size() == 2);
    CHECK(empty.get_Node(a_id).idx == a_id);
    CHECK(empty.get_Node(a_id).value == a);
    CHECK(empty.get_Node(a_id).level == 0);
    CHECK(empty.get_Node(b_id).neighbour.size() == 2);
    CHECK(throws_as<out_of_range>([&] { empty.get_Node(-1); }));
    CHECK(throws_as<out_of_range>([&] { empty.get_Node(empty.size()); }));

    // This test intentionally catches the dimension-check bug if add_node
    // does not throw for a vector whose dimension differs from existing nodes.

    HNSW greedy(4, 20);
    vector<float> g0{0, 0}, g1{2, 0}, g2{4, 0};
    int g0id = greedy.add_node(g0, 0);
    int g1id = greedy.add_node(g1, 0);
    int g2id = greedy.add_node(g2, 0);
    greedy.add_edge(g0id, g1id, 0);
    greedy.add_edge(g1id, g2id, 0);
    vector<float> qg{3.8f, 0};
    CHECK(greedy.greedy_search(qg, g0id, 0) == g2id);
    vector<float> qexact{2, 0};
    CHECK(greedy.greedy_search(qexact, g0id, 0) == g1id);
    CHECK(throws_as<out_of_range>([&] { greedy.greedy_search(qg, -1, 0); }));
    CHECK(throws_as<out_of_range>([&] { greedy.greedy_search(qg, g0id, 1); }));

    // search_layer: nearest candidates, ef boundaries and invalid arguments
    HNSW layer_graph(4, 20);
    vector<float> l0{0, 0}, l1{1, 0}, l2{2, 0}, l3{3, 0};
    int i0 = layer_graph.add_node(l0, 0);
    int i1 = layer_graph.add_node(l1, 0);
    int i2 = layer_graph.add_node(l2, 0);
    int i3 = layer_graph.add_node(l3, 0);
    layer_graph.add_edge(i0, i1, 0);
    layer_graph.add_edge(i1, i2, 0);
    layer_graph.add_edge(i2, i3, 0);
    vector<float> ql{2.1f, 0};
    auto result = layer_graph.search_layer(ql, i0, 3, 0);
    CHECK(!result.empty() && result.size() <= 3);
    CHECK(result.front() == i2);
    CHECK(layer_graph.search_layer(ql, i0, 1, 0).size() == 1);
    CHECK(layer_graph.search_layer(ql, i0, 100, 0).size() <= 4);
    CHECK(throws_as<invalid_argument>([&] { layer_graph.search_layer(ql, i0, 0, 0); }));
    CHECK(throws_as<out_of_range>([&] { layer_graph.search_layer(ql, -1, 2, 0); }));
    CHECK(throws_as<out_of_range>([&] { layer_graph.search_layer(ql, i0, 2, 1); }));

    // prune_neighbours: keep closest edges and remove reverse edge for removed node
    HNSW prune_graph(4, 20);
    vector<float> center{0, 0}, near1{1, 0}, near2{2, 0}, far3{3, 0};
    int ci = prune_graph.add_node(center, 0);
    int ni = prune_graph.add_node(near1, 0);
    int n2i = prune_graph.add_node(near2, 0);
    int fi = prune_graph.add_node(far3, 0);
    prune_graph.add_edge(ci, ni, 0);
    prune_graph.add_edge(ci, n2i, 0);
    prune_graph.add_edge(ci, fi, 0);
    prune_graph.prune_neighbours(ci, 0, 2);
    CHECK(prune_graph.get_Node(ci).neighbour[0].size() == 2);
    CHECK(contains(prune_graph.get_Node(ci).neighbour[0], ni));
    CHECK(contains(prune_graph.get_Node(ci).neighbour[0], n2i));
    CHECK(!contains(prune_graph.get_Node(fi).neighbour[0], ci));
    CHECK(throws_as<out_of_range>([&] { prune_graph.prune_neighbours(-1, 0, 2); }));
    CHECK(throws_as<out_of_range>([&] { prune_graph.prune_neighbours(ci, 1, 2); }));
    CHECK(throws_as<invalid_argument>([&] { prune_graph.prune_neighbours(ci, 0, 0); }));

    // insert: empty input, first point, IDs, dimension mismatch, multiple points
    HNSW inserted(4, 30);
    CHECK(throws_as<invalid_argument>([&] {
        vector<float> v;
        inserted.insert(v);
    }));
    vector<vector<float>> data{{0,0}, {1,0}, {2,0}, {3,0}, {0,2}, {4,0}, {5,0}, {2,2}};
    for (size_t i = 0; i < data.size(); ++i) {
        CHECK(inserted.insert(data[i]) == static_cast<int>(i));
        CHECK(inserted.size() == static_cast<int>(i + 1));
        CHECK(inserted.get_entry_point() >= 0 && inserted.get_entry_point() < inserted.size());
        CHECK(inserted.get_max_level() >= 0);
    }
    CHECK(throws_as<invalid_argument>([&] {
        vector<float> wrong{1, 2, 3};
        inserted.insert(wrong);
    }));

    // knn_search: nearest points, k/ef limits, empty index, dimension mismatch
    HNSW searchable(4, 30);
    CHECK(throws_as<invalid_argument>([&] {
        vector<float> q{0, 0};
        searchable.knn_search(q, 1, 10);
    }));
    vector<vector<float>> search_data{{0,0}, {1,0}, {2,0}, {3,0}, {0,2}, {4,0}};
    for (auto& point : search_data) searchable.insert(point);
    vector<float> q{2.1f, 0};
    auto knn = searchable.knn_search(q, 3, 20);
    CHECK(knn.size() == 3);
    CHECK(knn[0] == 2);
    CHECK(contains(knn, 1));
    CHECK(contains(knn, 3));
    CHECK(searchable.knn_search(q, 1, 1).size() == 1);
    CHECK(searchable.knn_search(q, searchable.size(), 100).size() == static_cast<size_t>(searchable.size()));
    CHECK(throws_as<invalid_argument>([&] { searchable.knn_search(q, 0, 10); }));
    CHECK(throws_as<invalid_argument>([&] { searchable.knn_search(q, searchable.size() + 1, 10); }));
    CHECK(throws_as<invalid_argument>([&] { searchable.knn_search(q, 1, 0); }));
    CHECK(throws_as<invalid_argument>([&] {
        vector<float> wrong{1, 2, 3};
        searchable.knn_search(wrong, 1, 10);
    }));

    // squared_distance and exact_search
    CHECK(abs(squared_distance(vector<float>{0,0}, vector<float>{3,4}) - 25.0f) < 1e-6f);
    CHECK(throws_as<invalid_argument>([] {
        squared_distance(vector<float>{1}, vector<float>{1,2});
    }));
    vector<vector<float>> exact_data{{0,0}, {1,0}, {2,0}, {4,0}};
    vector<float> qe{1.1f, 0};
    auto exact = exact_search(exact_data, qe, 3);
    CHECK(exact.size() == 3);
    CHECK(exact[0] == 1 && exact[1] == 2 && exact[2] == 0);
    CHECK(throws_as<invalid_argument>([&] { exact_search(exact_data, qe, 0); }));
    CHECK(throws_as<invalid_argument>([&] {
        vector<float> wrong{1, 2, 3};
        exact_search(exact_data, wrong, 2);
    }));

    cout << "All comprehensive HNSW tests passed. Checks run: " << checks << '\n';
    return 0;
}


