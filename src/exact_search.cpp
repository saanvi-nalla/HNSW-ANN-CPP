
#include "exact_search.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

using namespace std;

float squared_distance(const vector<float>& a,const vector<float>& b) {

    if (a.size() != b.size()) {
        throw std::invalid_argument("Dimension mismatch");
    }

    float distance = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        distance += diff * diff;
    }

    return distance;
}

vector<int> exact_search(const vector<vector<float>>& database,const vector<float>& query,int k) {
    
    if (k < 1 ||  static_cast<size_t>(k) > database.size()) {
        throw invalid_argument("Invalid k");
    }

    vector<pair<float, int>> distances;
    distances.reserve(database.size());

    for (int i = 0; i < static_cast<int>(database.size()); ++i) {
        float d = squared_distance(database[i], query);
        distances.push_back({d, i});
    }

    sort(distances.begin(), distances.end());

    vector<int> result;
    result.reserve(k);

    for (int i = 0; i < k; ++i) {
        result.push_back(distances[i].second);
    }

    return result;
}

