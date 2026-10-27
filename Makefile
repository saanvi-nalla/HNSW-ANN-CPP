
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic -O2
INCLUDES = -Iinclude

BUILD_DIR = build

LIB_SOURCES = src/hnsw_core.cpp src/hnsw_insert.cpp src/hnsw_search.cpp src/hnsw_neighbors.cpp src/exact_search.cpp

TEST_SOURCE = tests/test_all.cpp
BENCH_SOURCE = benchmarks/benchmark.cpp

.PHONY: all test benchmark clean

all: $(BUILD_DIR)/test_all $(BUILD_DIR)/benchmark

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_all: $(TEST_SOURCE) $(LIB_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TEST_SOURCE) $(LIB_SOURCES) -o $@

$(BUILD_DIR)/benchmark: $(BENCH_SOURCE) $(LIB_SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(BENCH_SOURCE) $(LIB_SOURCES) -o $@

test: $(BUILD_DIR)/test_all
	./$(BUILD_DIR)/test_all

benchmark: $(BUILD_DIR)/benchmark
	./$(BUILD_DIR)/benchmark

clean:
	rm -rf $(BUILD_DIR)
