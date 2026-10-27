
#ifndef EXACT_SEARCH_HPP
#define EXACT_SEARCH_HPP

#include <iostream>
#include <vector>

using namespace std;

float squared_distance(
    const vector<float>& a,
    const vector<float>& b
);

vector<int> exact_search(
    const vector<vector<float>>& database,
    const vector<float>& query,
    int k
);

#endif

