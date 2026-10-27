#ifndef HNSW_HPP
#define HNSW_HPP

#include <iostream>
#include <vector>
#include <queue>
#include <random> 
using namespace std;

struct Node{
	int idx;
	vector  <float> value;
	int level;
	vector <vector<int>> neighbour;
};

class HNSW{
	private:
	vector <Node> nodes;
	int entry_point;
	int max_level;
	int M;
	int ef_construction;
	double ml;
	mt19937_64 rng;
	//priority_queue <Node, vector<Node>, cmp> C;
	public:
	int size();
	int add_node(const vector <float>& value,int level);
	int get_entry_point();
	int get_max_level();
	int random_level();

	void add_edge(int a,int b, int layer);
	Node& get_Node(int idx);
	int greedy_search(vector <float> query, int entry_point,int level);
	void prune_neighbours(int node_id,int layer, int max_degree);
	int insert(vector <float>& value);
	explicit HNSW(int max_connections = 16, int construction_ef = 200);

	vector <int> knn_search (vector <float>& query, int k, int ef_search = 50);
	vector <int> search_layer(vector <float>& query,int entry_point, int ef, int layer);
	vector <int> search_neighbor(const vector<float>& query, const vector <int>& candidates, int max_degree);
};
#endif
