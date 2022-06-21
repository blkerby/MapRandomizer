#include <inttypes.h>
#include <iostream>
#include <vector>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

typedef uint16_t vertex_t;
typedef uint8_t component_t;
typedef int64_t successor_set_t;
typedef std::vector<std::vector<vertex_t>> adjacency_t;
typedef std::vector<std::unordered_set<component_t>> cond_adjacency_t;

// Tarjan's algorithm for computing strong components
vertex_t tarjan_rec(
        vertex_t root, 
        vertex_t& next_vertex,
        component_t& next_component,
        const adjacency_t& adjacency, 
        std::vector<vertex_t>& visited_num,
        std::vector<bool>& on_stack,
        std::vector<vertex_t>& stack,
        component_t* output_components) {
    vertex_t current_visited_num = next_vertex;
    vertex_t lowval = next_vertex;

    visited_num[root] = next_vertex;
    on_stack[root] = true;
    next_vertex++;
    stack.push_back(root);
    for (vertex_t child: adjacency[root]) {
        if (visited_num[child] == 0) {  // child not yet visited
            vertex_t child_lowval = tarjan_rec(
                child, next_vertex, next_component, adjacency, visited_num, on_stack, stack, output_components);
            lowval = std::min(lowval, child_lowval);
        } else if (on_stack[child]) {
            lowval = std::min(lowval, visited_num[child]);
        }
    }
    if (lowval == current_visited_num) {
        while (true) {
            vertex_t v = stack.back();
            stack.pop_back();
            on_stack[v] = false;
            output_components[v] = next_component;
            if (v == root) break;
        }
        next_component++;
    }
    return lowval;
}

// Compute a compressed representation of reflexive-transitive closure of the given directed graphs. More specifically,
// compute the strong components of each graph and the adjacency matrix of the reflexive-transitive closure of
// the condensation graph.
void compute_connectivity(
        py::array_t<bool> root_mask,
        py::array_t<vertex_t> directed_edges,
        py::array_t<vertex_t> undirected_edges,
        py::array_t<int32_t> undirected_boundaries,
        py::array_t<component_t> output_components,
        py::array_t<successor_set_t> output_adjacency) {
    int num_graphs = root_mask.shape(0);
    int num_vertices = root_mask.shape(1);
    auto root_mask_proxy = root_mask.unchecked<2>();
    auto directed_edges_proxy = directed_edges.unchecked<2>();
    auto undirected_edges_proxy = undirected_edges.unchecked<2>();
    auto undirected_boundaries_ptr = undirected_boundaries.data();
    for (int g = 0; g < num_graphs; g++) {
        adjacency_t adjacency;

        for (int i = 0; i < num_vertices; i++){
            adjacency.push_back(std::vector<vertex_t>());
        }

        for (int i = 0; i < directed_edges_proxy.shape(0); i++) {
            vertex_t src = *directed_edges_proxy.data(i, 0);
            vertex_t dst = *directed_edges_proxy.data(i, 1);
            adjacency[src].push_back(dst);
        }

        int32_t ue_start = undirected_boundaries_ptr[g];
        int32_t ue_end = (g == num_graphs - 1) ? undirected_edges_proxy.shape(0) : undirected_boundaries_ptr[g + 1];
        for (int i = ue_start; i < ue_end; i++) {
            vertex_t src = *undirected_edges_proxy.data(i, 0);
            vertex_t dst = *undirected_edges_proxy.data(i, 1);
            adjacency[src].push_back(dst);
            adjacency[dst].push_back(src);
        }

        std::vector<vertex_t> visited_num(num_vertices);
        std::vector<bool> on_stack(num_vertices);
        std::vector<vertex_t> stack;
        vertex_t next_vertex = 1;
        component_t next_component = 1;
        for (int i = 0; i < num_vertices; i++) {
            if (*root_mask_proxy.data(g, i) && visited_num[i] == 0) {
                tarjan_rec(i, next_vertex, next_component, adjacency, visited_num, on_stack, stack,
                output_components.mutable_data(g, 0));
            }
        }

    //    std::cout << "next_component=" << (int)next_component << std::endl;

        // Construct the condensation graph
        cond_adjacency_t condensation_adjacency;
        for (int i = 0; i < next_component; i++){
            condensation_adjacency.push_back(std::unordered_set<component_t>());
        }
        auto output_components_ptr = output_components.data(g, 0);
        auto output_adjacency_ptr = output_adjacency.mutable_data(g, 0);
        for (int i = 0; i < directed_edges_proxy.shape(0); i++) {
            vertex_t src = *directed_edges_proxy.data(i, 0);
            vertex_t dst = *directed_edges_proxy.data(i, 1);
            component_t src_comp = output_components_ptr[src];
            component_t dst_comp = output_components_ptr[dst];
            condensation_adjacency[src_comp].insert(dst_comp);
    //        std::cout << (int)src << " -> " << (int)dst << " : " << (int)src_comp << " -> " << (int)dst_comp << std::endl;
        }
        // Note: there is no need to consider the undirected edges, since vertices joined by an undirected edge would
        // already be contracted to a single vertex in the condensation graph

        // Compute the reflexive-transitive closure of the condensation graph
        for (int i = 1; i < next_component; i++) {
            successor_set_t successor = 1L << i;  // Ensure the relation is reflexive
    //        std::cout << "start component " << i << std::endl;
            for (component_t child: condensation_adjacency[i]) {
    //            std::cout << "child " << (int)child << ": " << output_adjacency_ptr[child] << std::endl;
                successor |= output_adjacency_ptr[child];
            }
    //        std::cout << "end component " << i << ": " << successor << std::endl;
            output_adjacency_ptr[i] = successor;
        }
    }
}

void compute_connectivity2(
        py::array_t<bool> room_mask,
        py::array_t<uint8_t> room_position_x,
        py::array_t<uint8_t> room_position_y,
        py::array_t<uint8_t> room_left,
        py::array_t<uint8_t> room_right,
        py::array_t<uint8_t> room_up,
        py::array_t<uint8_t> room_down,
        py::array_t<int64_t> part_left,
        py::array_t<int64_t> part_right,
        py::array_t<int64_t> part_up,
        py::array_t<int64_t> part_down,
        py::array_t<int64_t> part_room_id,
        py::array_t<int64_t> good_parts,
        int64_t map_x,
        int64_t map_y,
        int64_t num_parts,
        py::array_t<vertex_t> directed_edges,
        py::array_t<component_t> output_components,
        py::array_t<successor_set_t> output_adjacency,
        py::array_t<float> output_adjacency_unpacked
        ) {
    auto room_mask_proxy = room_mask.unchecked<2>();
    auto room_position_x_proxy = room_position_x.unchecked<2>();
    auto room_position_y_proxy = room_position_y.unchecked<2>();
    auto room_left_proxy = room_left.unchecked<2>();
    auto room_right_proxy = room_right.unchecked<2>();
    auto room_up_proxy = room_up.unchecked<2>();
    auto room_down_proxy = room_down.unchecked<2>();
    auto part_left_proxy = part_left.unchecked<1>();
    auto part_right_proxy = part_right.unchecked<1>();
    auto part_up_proxy = part_up.unchecked<1>();
    auto part_down_proxy = part_down.unchecked<1>();
    auto part_room_id_proxy = part_room_id.unchecked<1>();
    auto good_parts_proxy = good_parts.unchecked<1>();
    auto directed_edges_proxy = directed_edges.unchecked<2>();
    auto output_components_proxy = output_components.mutable_unchecked<2>();
    auto output_adjacency_proxy = output_adjacency.mutable_unchecked<2>();
    auto output_adjacency_unpacked_proxy = output_adjacency_unpacked.mutable_unchecked<3>();
    int num_graphs = room_mask_proxy.shape(0);
    std::unordered_map<uint16_t, vertex_t> position_to_part_map;
    adjacency_t adjacency;
    for (int i = 0; i < num_parts; i++){
        adjacency.push_back(std::vector<vertex_t>());
    }

    for (int g = 0; g < num_graphs; g++) {
        for (int i = 0; i < num_parts; i++){
            adjacency[i].clear();
        }

        // Populate undirected edges for horizontal doors
        position_to_part_map.clear();
        for (int i = 0; i < room_left_proxy.shape(0); i++) {
            int room_id = *room_left_proxy.data(i, 0);
            int mask = *room_mask_proxy.data(g, room_id);
            if (!mask) {
                continue;
            }
            int rel_x = *room_left_proxy.data(i, 1);
            int rel_y = *room_left_proxy.data(i, 2);
            int part = *part_left_proxy.data(i);
            int room_x = *room_position_x_proxy.data(g, room_id);
            int room_y = *room_position_y_proxy.data(g, room_id);
            int x = room_x + rel_x;
            int y = room_y + rel_y;
            int pos = (y << 8) + x;
            position_to_part_map[pos] = part;
        }
        for (int i = 0; i < room_right_proxy.shape(0); i++) {
            int room_id = *room_right_proxy.data(i, 0);
            int mask = *room_mask_proxy.data(g, room_id);
            if (!mask) {
                continue;
            }
            int rel_x = *room_right_proxy.data(i, 1);
            int rel_y = *room_right_proxy.data(i, 2);
            int part = *part_right_proxy.data(i);
            int room_x = *room_position_x_proxy.data(g, room_id);
            int room_y = *room_position_y_proxy.data(g, room_id);
            int x = room_x + rel_x;
            int y = room_y + rel_y;
            int pos = (y << 8) + x;
            auto it = position_to_part_map.find(pos);
            if (it != position_to_part_map.end()) {
                int other_part = it->second;
                adjacency[other_part].push_back(part);
                adjacency[part].push_back(other_part);
            }
        }

        // Populate undirected edges for vertical doors
        position_to_part_map.clear();
        for (int i = 0; i < room_up_proxy.shape(0); i++) {
            int room_id = *room_up_proxy.data(i, 0);
            int mask = *room_mask_proxy.data(g, room_id);
            if (!mask) {
                continue;
            }
            int rel_x = *room_up_proxy.data(i, 1);
            int rel_y = *room_up_proxy.data(i, 2);
            int part = *part_up_proxy.data(i);
            int room_x = *room_position_x_proxy.data(g, room_id);
            int room_y = *room_position_y_proxy.data(g, room_id);
            int x = room_x + rel_x;
            int y = room_y + rel_y;
            int pos = (y << 8) + x;
            position_to_part_map[pos] = part;
        }
        for (int i = 0; i < room_down_proxy.shape(0); i++) {
            int room_id = *room_down_proxy.data(i, 0);
            int mask = *room_mask_proxy.data(g, room_id);
            if (!mask) {
                continue;
            }
            int rel_x = *room_down_proxy.data(i, 1);
            int rel_y = *room_down_proxy.data(i, 2);
            int part = *part_down_proxy.data(i);
            int room_x = *room_position_x_proxy.data(g, room_id);
            int room_y = *room_position_y_proxy.data(g, room_id);
            int x = room_x + rel_x;
            int y = room_y + rel_y;
            int pos = (y << 8) + x;
            auto it = position_to_part_map.find(pos);
            if (it != position_to_part_map.end()) {
                int other_part = it->second;
                adjacency[other_part].push_back(part);
                adjacency[part].push_back(other_part);
            }
        }

        // Populate directed edges
        for (int i = 0; i < directed_edges_proxy.shape(0); i++) {
            vertex_t src = *directed_edges_proxy.data(i, 0);
            vertex_t dst = *directed_edges_proxy.data(i, 1);
            adjacency[src].push_back(dst);
        }

        // Compute the strong components
        std::vector<vertex_t> visited_num(num_parts);
        std::vector<bool> on_stack(num_parts);
        std::vector<vertex_t> stack;
        vertex_t next_vertex = 1;
        component_t next_component = 1;
        for (int i = 0; i < num_parts; i++) {
            if (visited_num[i] == 0) {
                tarjan_rec(i, next_vertex, next_component, adjacency, visited_num, on_stack, stack,
                    output_components_proxy.mutable_data(g, 0));
            }
        }

        // Construct the condensation graph
        cond_adjacency_t condensation_adjacency;
        for (int i = 0; i < next_component; i++){
            condensation_adjacency.push_back(std::unordered_set<component_t>());
        }
        auto output_components_ptr = output_components_proxy.data(g, 0);
        auto output_adjacency_ptr = output_adjacency_proxy.mutable_data(g, 0);
        for (int i = 0; i < directed_edges_proxy.shape(0); i++) {
            vertex_t src = *directed_edges_proxy.data(i, 0);
            vertex_t dst = *directed_edges_proxy.data(i, 1);
            component_t src_comp = output_components_ptr[src];
            component_t dst_comp = output_components_ptr[dst];
            condensation_adjacency[src_comp].insert(dst_comp);
        }

        // Compute the reflexive-transitive closure of the condensation graph
        for (int i = 1; i < next_component; i++) {
            successor_set_t successor = 1L << i;  // Ensure the relation is reflexive
            for (component_t child: condensation_adjacency[i]) {
                successor |= output_adjacency_ptr[child];
            }
            output_adjacency_ptr[i] = successor;
            for (int j = 0; j < next_component; j++) {
                if (successor & (1L << j)) {
                    *output_adjacency_unpacked_proxy.mutable_data(g, i, j) = 1.0;
                }
            }
        }
    }
}

PYBIND11_MODULE(connectivity, m) {
    m.def("compute_connectivity", &compute_connectivity,
          "compute a compressed representation of transitive closure of directed graphs");
    m.def("compute_connectivity2", &compute_connectivity2,
          "compute a compressed representation of transitive closure of directed graphs");
}