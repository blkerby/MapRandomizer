from rando.sm_json_data import SMJsonData, GameState, DifficultyConfig

def compute_escape_data(map, sm_json_data: SMJsonData, difficulty: DifficultyConfig, door_edges):
    state = GameState(
        items=sm_json_data.item_set,   # All items collected
        flags=sm_json_data.flags_set,  # All flags set
        weapons=sm_json_data.get_weapons(sm_json_data.item_set),
        num_energy_tanks=14,
        num_reserves=4,
        max_energy=1800,
        max_missiles=230,
        max_super_missiles=50,
        max_power_bombs=50,
        current_energy=1800,
        current_missiles=230,
        current_super_missiles=50,
        current_power_bombs=50,
        vertex_index=sm_json_data.vertex_index_dict[(238, 1, 0)])  # Mother Brain room left door
    reach, route_data = sm_json_data.compute_reachable_vertices(state, difficulty, door_edges)
    spoiler_data = sm_json_data.get_spoiler_entry(
        sm_json_data.vertex_index_dict[(8, 5, 0)],  # Ship (Landing Site)
        route_data, state, state, '', 0, 0, map)

    pass