#include "hnsw.hpp"
#include "exact_search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

int main() {
    // Dataset configuration
    const int N = 3000;
    const int DIM = 32;
    const int NUM_QUERIES = 100;
    const int K = 10;

    // HNSW configuration
    const int M = 16;
    const int EF_CONSTRUCTION = 100;
    const int EF_SEARCH = 100;

    // Fixed seed makes experiments reproducible.
    mt19937 rng(42);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    vector<vector<float>> database(N, vector<float>(DIM));
    vector<vector<float>> queries(
        NUM_QUERIES, vector<float>(DIM)
    );

    for (auto& point : database) {
        for (float& x : point) {
            x = dist(rng);
        }
    }

    for (auto& query : queries) {
        for (float& x : query) {
            x = dist(rng);
        }
    }

    // Build the HNSW index.
    HNSW graph(M, EF_CONSTRUCTION);

    auto build_start = Clock::now();

    for (auto& point : database) {
        graph.insert(point);
    }

    auto build_end = Clock::now();

    double build_ms =
        chrono::duration<double, milli>(
            build_end - build_start
        ).count();

    vector<double> latencies_us;
    latencies_us.reserve(NUM_QUERIES);

    long long total_hits = 0;
    long long total_exact_results = 0;

    // Run queries and compare against exact search.
    for (auto& query : queries) {
        auto start = Clock::now();

        vector<int> approximate =
            graph.knn_search(query, K, EF_SEARCH);

        auto end = Clock::now();

        double latency_us =
            chrono::duration<double, micro>(
                end - start
            ).count();

        latencies_us.push_back(latency_us);

        // Ground truth from brute-force search.
        vector<int> exact =
            exact_search(database, query, K);

        // Count approximate results present in ground truth.
        for (int id : approximate) {
            if (find(exact.begin(), exact.end(), id)
                != exact.end()) {
                ++total_hits;
            }
        }

        total_exact_results += exact.size();
    }

    // Sort latencies to calculate percentiles.
    sort(latencies_us.begin(), latencies_us.end());

    double recall =
        total_exact_results == 0
            ? 0.0
            : static_cast<double>(total_hits)
                  / total_exact_results;

    double median_us =
        latencies_us[latencies_us.size() / 2];

    size_t p95_index =
        static_cast<size_t>(
            ceil(0.95 * latencies_us.size())
        ) - 1;

    double p95_us = latencies_us[p95_index];

    double total_query_ms = 0.0;
    for (double latency : latencies_us) {
        total_query_ms += latency / 1000.0;
    }

    double qps =
        total_query_ms == 0.0
            ? 0.0
            : NUM_QUERIES * 1000.0 / total_query_ms;

    cout << "========== HNSW Benchmark ==========\n";
    cout << "Dataset size       : " << N << '\n';
    cout << "Dimension          : " << DIM << '\n';
    cout << "Queries            : " << NUM_QUERIES << '\n';
    cout << "K                  : " << K << '\n';
    cout << "M                  : " << M << '\n';
    cout << "ef_construction    : "
         << EF_CONSTRUCTION << '\n';
    cout << "ef_search          : " << EF_SEARCH << '\n';

    cout << "\nBuild time         : "
         << build_ms << " ms\n";

    cout << "Recall@" << K << "          : "
         << recall << '\n';

    cout << "Median latency     : "
         << median_us << " us\n";

    cout << "P95 latency        : "
         << p95_us << " us\n";

    cout << "Average QPS        : "
         << qps << '\n';

    cout << "====================================\n";

    return 0;
}