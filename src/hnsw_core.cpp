#include "hnsw.hpp"
#include "exact_search.hpp"

#include <algorithm>
#include <utility>
#include <stdexcept>
#include <cmath> 
#include <random>

HNSW::HNSW(int max_connections, int construction_ef)
    : entry_point(-1),
      max_level(-1),
      M(max_connections),
      ef_construction(construction_ef),
      ml(1.0 / log(static_cast<double>(max_connections))),
      rng(random_device{}()) {
    if (M < 2)
        throw invalid_argument("M must be at least 2");

    if (ef_construction < 1)
        throw invalid_argument("ef_construction must be positive");
}

int HNSW:: add_node(const vector <float>& value, int level){

    if(level<0)
    throw invalid_argument("Level cannot be negative");

    if(!nodes.empty() && value.size() != nodes[0].value.size())
    throw invalid_argument("Dimension mismatch");
    Node node;
    node.idx = nodes.size();
    node.value = value;
    node.level = level;
    node.neighbour.resize(level+1);
    nodes.push_back(node);

    return node.idx;
}

Node& HNSW:: get_Node(int idx){
     if (idx < 0 || idx >= size())
    throw out_of_range("Invalid node index");

    return nodes[idx];
}

int HNSW:: size(){
    return nodes.size();
}

int HNSW::get_entry_point() {
    return entry_point;
}

int HNSW::get_max_level() {
    return max_level;
}

void HNSW:: add_edge(int a,int b, int layer){

    if(a<0 || b<0 || a>=size() || b>=size()){
        cout<<"INVALID INDICES"<<endl;
        return;
    }

    if(layer<0 || layer>nodes[a].level || layer>nodes[b].level){
        cout<<"INVALID INDICES"<<endl;
        return;
    }
    if(a==b)
    return;
    
    auto &a_node = nodes[a].neighbour[layer];
    auto &b_node = nodes[b].neighbour[layer];

    if (find(a_node.begin(), a_node.end(), b) == a_node.end())
    a_node.push_back(b);

    if (find(b_node.begin(), b_node.end(), a) == b_node.end())
    b_node.push_back(a);

    return;
}

int HNSW:: random_level(){
    double u;
    do{
        u = generate_canonical<double, 53>(rng);
    }while(u<=0);

    return (-log(u)*ml);
}
