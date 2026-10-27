# HNSW: Approximate Nearest Neighbor Search in C++

A C++17 implementation of **Hierarchical Navigable Small World (HNSW)**, a graph-based algorithm for efficient approximate nearest neighbor search. This project explores how hierarchical graphs can accelerate vector search while balancing search speed, accuracy, and memory usage.

**Project status:** Research and implementation in progress.

## Research Paper

This project is based on the research paper:

**Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs**

*Yu. A. Malkov and D. A. Yashunin*

[Read the paper on arXiv](https://arxiv.org/abs/1603.09320)

## How HNSW Works

Traditional exact nearest neighbor search calculates the distance between a query and every vector in the database. Although accurate, this becomes expensive as the dataset grows.

HNSW reduces the search effort by organizing vectors into a hierarchy of graphs:

- **Hierarchical structure:** Nodes are assigned random levels, creating progressively smaller upper layers.
- **Greedy navigation:** Search starts at an upper layer and moves towards closer nodes.
- **Layer-wise search:** The algorithm descends through the hierarchy, exploring more candidates at the lower layers.
- **Neighbor selection:** During insertion, candidate neighbors are selected and connections are pruned to control graph connectivity.

The result is an approximate search algorithm that can reduce query time while maintaining high recall, depending on the dataset and parameter choices.

## Implementation and Research Mapping

| Research concept | Implementation |
|---|---|
| Hierarchical graph construction | Node levels and per-layer neighbor lists |
| Random level assignment | `random_level()` |
| Incremental insertion | `insert()` |
| Greedy navigation | `greedy_search()` |
| Candidate exploration | `search_layer()` |
| Neighbor selection and pruning | `search_neighbour()` and `prune_neighbours()` |
| Approximate nearest neighbor queries | `knn_search()` |
| Exact-search baseline | `exact_search()` |

The implementation separates graph management, insertion, searching, and neighbor selection into different source files. This makes the algorithm easier to test, understand, and optimize.

Neighbor-selection logic and other implementation details are being validated against the original paper; the project does not assume complete equivalence with every detail of the published algorithm.

## Project Structure

```text
include/
    hnsw.hpp
    exact_search.hpp
src/
    hnsw_core.cpp
    hnsw_insert.cpp
    hnsw_search.cpp
    hnsw_neighbors.cpp
    exact_search.cpp
tests/
    test_all.cpp
benchmarks/
    benchmark.cpp
Makefile
CMakeLists.txt
README.md
```

## Getting Started

**Requirements:** GCC or another C++17-compatible compiler, GNU Make, and Git.

Build the project:

```bash
make
```

Run the tests:

```bash
make test
```

Run the benchmark:

```bash
make benchmark
```

Remove generated build files:

```bash
make clean
```

## Benchmarking and Evaluation

The project includes an exact-search implementation to serve as a reference for evaluating approximate results.

The planned evaluation focuses on:

- **Construction time:** Time required to build the graph.
- **Query latency:** Median and 95th-percentile response time.
- **Recall@k:** Fraction of exact nearest neighbors recovered by HNSW.
- **Throughput:** Number of queries processed per second.
- **Parameter tuning:** Studying the effects of `M`, `ef_construction`, and `ef_search`.

These measurements will help characterize the trade-off between search quality and computational cost. Performance conclusions will be based on reproducible experiments rather than assumed improvements.

## Future Work

- Validate graph construction and neighbor selection against the paper.
- Expand correctness tests and graph-invariant checks.
- Benchmark different datasets and parameter configurations.
- Profile memory access and distance calculations.
- Explore SIMD vectorization and parallelization where beneficial.

## Reference

Malkov, Y. A., and Yashunin, D. (2020). *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4), 824–836.

[Paper](https://arxiv.org/abs/1603.09320) · [DOI](https://doi.org/10.1109/TPAMI.2018.2889473)

---

*An independent educational implementation inspired by the published HNSW research.*
