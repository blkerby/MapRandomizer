#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cassert>
#include <unordered_set>
#include <iostream>

namespace py = pybind11;

static constexpr int NUM_RESOURCE_TYPES = 4;
using vertex_t = int16_t;
using cost_t = int16_t;

struct Edge {
    int16_t id;
    vertex_t dst;
    cost_t resource_cost[NUM_RESOURCE_TYPES];
};

struct Value {
    float heuristic_value;
    cost_t resource_quantity[NUM_RESOURCE_TYPES];
};

using Graph = std::vector<std::vector<Edge>>;

inline float compute_heuristic_value(
    cost_t resource_quantity[NUM_RESOURCE_TYPES],
    cost_t max_resources[NUM_RESOURCE_TYPES]
) {
    float total_value = 0.0;
    float eps = 1e-15;
    for (int i = 0; i < NUM_RESOURCE_TYPES; i++) {
        total_value += static_cast<float>(resource_quantity[i]) / (static_cast<float>(max_resources[i]) + eps);
    }
    return total_value;
}

// Variation of Bellman-Ford algorithm
void compute_reachability(py::array_t<int16_t> graph_input,
                         vertex_t start_vertex,
                         vertex_t num_vertices,
                         bool track_route,
                         py::array_t<cost_t> start_resources_input,
                         py::array_t<cost_t> max_resources_input,
                         py::array_t<cost_t> output_cost,
                         py::array_t<int32_t> output_route_id,
                         py::array_t<int16_t> output_route_edge,
                         py::array_t<int32_t> output_route_prev) {
    auto graph_input_uc = graph_input.unchecked<2>();
    auto start_resources_input_uc = start_resources_input.unchecked<1>();
    auto max_resources_input_uc = max_resources_input.unchecked<1>();
    auto output_cost_uc = output_cost.mutable_unchecked<2>();
    auto output_route_id_uc = output_route_id.mutable_unchecked<1>();
    auto output_route_edge_uc = output_route_edge.mutable_unchecked<1>();
    auto output_route_prev_uc = output_route_prev.mutable_unchecked<1>();
    int output_route_size = output_route_edge.size();
    int output_route_next = 1;
    assert(graph_input_uc.shape(1) >= NUM_RESOURCE_TYPES + 2);
    assert(start_resources_input_uc.shape(0) == NUM_RESOURCE_TYPES);
    assert(max_resources_input_uc.shape(0) == NUM_RESOURCE_TYPES);
    assert(start_vertex < num_vertices);
    assert(start_vertex >= 0);
    assert(output_cost_uc.shape(0) == num_vertices);
    assert(output_cost_uc.shape(1) == NUM_RESOURCE_TYPES);
    assert(output_route_id_uc.shape(0) == num_vertices);
    assert(output_route_prev_uc.shape(0) == output_route_edge.shape(0));

    cost_t start_resources[NUM_RESOURCE_TYPES];
    cost_t max_resources[NUM_RESOURCE_TYPES];
    for (int i = 0; i < NUM_RESOURCE_TYPES; i++) {
        start_resources[i] = start_resources_input_uc(i);
        max_resources[i] = max_resources_input_uc(i);
    }

    // Construct graph as list of outgoing-edge lists by vertex.
    Graph graph;
    graph.resize(num_vertices);
    int n = graph_input_uc.shape(0);
    for (int i = 0; i < n; i++) {
        int src = graph_input_uc(i, 0);
        assert(src >= 0);
        assert(src < num_vertices);

        int dst = graph_input_uc(i, 1);
        assert(dst >= 0);
        assert(dst < num_vertices);

        Edge e;
        e.id = i;
        e.dst = dst;
        for (int j = 0; j < NUM_RESOURCE_TYPES; j++) {
            e.resource_cost[j] = graph_input_uc(i, j + 2);
        }
        graph[src].push_back(e);
    }

    Value initial_value;
    for (int i = 0; i < NUM_RESOURCE_TYPES; i++) {
        initial_value.resource_quantity[i] = -1;
    }
    initial_value.heuristic_value = compute_heuristic_value(initial_value.resource_quantity, max_resources);
    std::vector<Value> vertex_best_value;
    vertex_best_value.resize(num_vertices, initial_value);
    for (int i = 0; i < NUM_RESOURCE_TYPES; i++) {
        vertex_best_value[start_vertex].resource_quantity[i] = start_resources[i];
    }
    vertex_best_value[start_vertex].heuristic_value = compute_heuristic_value(
        start_resources, max_resources);

    std::unordered_set<vertex_t> modified_vertices;
    modified_vertices.insert(start_vertex);

    std::unordered_set<vertex_t> new_modified_vertices;
    while (!modified_vertices.empty()) {
        for (vertex_t src : modified_vertices) {
            Value src_value = vertex_best_value[src];
            for (Edge edge : graph[src]) {
                Value dst_value;
                for (int i = 0; i < NUM_RESOURCE_TYPES; i++){
                    dst_value.resource_quantity[i] = src_value.resource_quantity[i] - edge.resource_cost[i];
                    if (dst_value.resource_quantity[i] < 0) {
                        goto impossible_edge;
                    }
                    if (dst_value.resource_quantity[i] > max_resources[i]) {
                        dst_value.resource_quantity[i] = max_resources[i];
                    }
                }
                dst_value.heuristic_value = compute_heuristic_value(dst_value.resource_quantity, max_resources);
                if (dst_value.heuristic_value > vertex_best_value[edge.dst].heuristic_value) {
                    vertex_best_value[edge.dst] = dst_value;
                    new_modified_vertices.insert(edge.dst);
                    if (track_route) {
                        // Update route info (for extracting spoiler log)
                        output_route_prev_uc(output_route_next) = output_route_id_uc(src);
                        output_route_edge_uc(output_route_next) = edge.id;
                        output_route_id_uc(edge.dst) = output_route_next;
                        output_route_next++;
                        assert(output_route_next < output_route_size);
                    }
                }
            impossible_edge:;
            }
        }
        modified_vertices.swap(new_modified_vertices);
        new_modified_vertices.clear();
    }

    for (int i = 0; i < num_vertices; i++) {
        for (int j = 0; j < NUM_RESOURCE_TYPES; j++) {
            output_cost_uc(i, j) += vertex_best_value[i].resource_quantity[j];
        }
    }
}

PYBIND11_MODULE(reachability, m) {
    m.def("compute_reachability", &compute_reachability,
        "Conservatively estimate the set of vertices reachable from starting vertex with given resources");
}