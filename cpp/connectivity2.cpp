#include <torch/extension.h>
#include <inttypes.h>
#include <iostream>
#include <vector>
#include <unordered_map>

typedef int16_t vertex_t;
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


void compute_connectivity2(
        torch::Tensor room_mask,
        torch::Tensor room_position_x,
        torch::Tensor room_position_y,
        torch::Tensor room_left,
        torch::Tensor room_right,
        torch::Tensor room_up,
        torch::Tensor room_down,
        torch::Tensor part_left,
        torch::Tensor part_right,
        torch::Tensor part_up,
        torch::Tensor part_down,
        torch::Tensor part_room_id,
        int64_t map_x,
        int64_t map_y,
        int64_t num_parts,
        torch::Tensor directed_edges,
        torch::Tensor output_components,
        torch::Tensor output_adjacency,
        torch::Tensor output_adjacency_unpacked
        ) {
    auto room_mask_proxy = room_mask.accessor<bool, 2>();
    auto room_position_x_proxy = room_position_x.accessor<int64_t, 2>();
    auto room_position_y_proxy = room_position_y.accessor<int64_t, 2>();
    auto room_left_proxy = room_left.accessor<int64_t, 2>();
    auto room_right_proxy = room_right.accessor<int64_t, 2>();
    auto room_up_proxy = room_up.accessor<int64_t, 2>();
    auto room_down_proxy = room_down.accessor<int64_t, 2>();
    auto part_left_proxy = part_left.accessor<int64_t, 1>();
    auto part_right_proxy = part_right.accessor<int64_t, 1>();
    auto part_up_proxy = part_up.accessor<int64_t, 1>();
    auto part_down_proxy = part_down.accessor<int64_t, 1>();
    auto part_room_id_proxy = part_room_id.accessor<int64_t, 1>();
    auto directed_edges_proxy = directed_edges.accessor<vertex_t, 2>();
    auto output_components_proxy = output_components.accessor<component_t, 2>();
    auto output_adjacency_proxy = output_adjacency.accessor<successor_set_t, 2>();
    auto output_adjacency_unpacked_proxy = output_adjacency_unpacked.accessor<float, 3>();
    int num_graphs = room_mask_proxy.size(0);

    at::parallel_for(0, num_graphs, 0, [&](int64_t start, int64_t end) {
        std::unordered_map<uint16_t, vertex_t> position_to_part_map;
        adjacency_t adjacency;
        for (int i = 0; i < num_parts; i++){
            adjacency.push_back(std::vector<vertex_t>());
        }
        for (int g = start; g < end; g++) {
            for (int i = 0; i < num_parts; i++){
                adjacency[i].clear();
            }

            // Populate undirected edges for horizontal doors
            position_to_part_map.clear();
            for (int i = 0; i < room_left_proxy.size(0); i++) {
                int room_id = room_left_proxy[i][0];
                int mask = room_mask_proxy[g][room_id];
                if (!mask) {
                    continue;
                }
                int rel_x = room_left_proxy[i][1];
                int rel_y = room_left_proxy[i][2];
                int part = part_left_proxy[i];
                int room_x = room_position_x_proxy[g][room_id];
                int room_y = room_position_y_proxy[g][room_id];
                int x = room_x + rel_x;
                int y = room_y + rel_y;
                int pos = (y << 8) + x;
                position_to_part_map[pos] = part;
            }
            for (int i = 0; i < room_right_proxy.size(0); i++) {
                int room_id = room_right_proxy[i][0];
                int mask = room_mask_proxy[g][room_id];
                if (!mask) {
                    continue;
                }
                int rel_x = room_right_proxy[i][1];
                int rel_y = room_right_proxy[i][2];
                int part = part_right_proxy[i];
                int room_x = room_position_x_proxy[g][room_id];
                int room_y = room_position_y_proxy[g][room_id];
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
            for (int i = 0; i < room_up_proxy.size(0); i++) {
                int room_id = room_up_proxy[i][0];
                int mask = room_mask_proxy[g][room_id];
                if (!mask) {
                    continue;
                }
                int rel_x = room_up_proxy[i][1];
                int rel_y = room_up_proxy[i][2];
                int part = part_up_proxy[i];
                int room_x = room_position_x_proxy[g][room_id];
                int room_y = room_position_y_proxy[g][room_id];
                int x = room_x + rel_x;
                int y = room_y + rel_y;
                int pos = (y << 8) + x;
                position_to_part_map[pos] = part;
            }
            for (int i = 0; i < room_down_proxy.size(0); i++) {
                int room_id = room_down_proxy[i][0];
                int mask = room_mask_proxy[g][room_id];
                if (!mask) {
                    continue;
                }
                int rel_x = room_down_proxy[i][1];
                int rel_y = room_down_proxy[i][2];
                int part = part_down_proxy[i];
                int room_x = room_position_x_proxy[g][room_id];
                int room_y = room_position_y_proxy[g][room_id];
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
            for (int i = 0; i < directed_edges_proxy.size(0); i++) {
                vertex_t src = directed_edges_proxy[i][0];
                vertex_t dst = directed_edges_proxy[i][1];
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
                        &output_components_proxy[g][0]);
                }
            }

            // Construct the condensation graph
            cond_adjacency_t condensation_adjacency;
            for (int i = 0; i < next_component; i++){
                condensation_adjacency.push_back(std::unordered_set<component_t>());
            }
            auto output_components_ptr = &output_components_proxy[g][0];
            auto output_adjacency_ptr = &output_adjacency_proxy[g][0];
            for (int i = 0; i < directed_edges_proxy.size(0); i++) {
                vertex_t src = directed_edges_proxy[i][0];
                vertex_t dst = directed_edges_proxy[i][1];
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
                        output_adjacency_unpacked_proxy[g][i][j] = 1.0;
                    }
                }
            }
        }
    });
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_connectivity2", &compute_connectivity2,
          "compute a compressed representation of transitive closure of directed graphs");
}
