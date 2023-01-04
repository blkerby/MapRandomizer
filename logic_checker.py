import streamlit as st
from rando.sm_json_data import SMJsonData, DifficultyConfig, Consumption, GameState
from functools import partial
from collections import defaultdict
import pathlib
import pandas as pd

st.set_page_config(layout="wide")

sm_json_data_path = "sm-json-data/"


@partial(st.cache, allow_output_mutation=True)
def load_sm_json_data():
    sm_json_data = SMJsonData(sm_json_data_path)
    return sm_json_data


sm_json_data = load_sm_json_data()

for tech in sm_json_data.tech_name_set:
    if not tech.startswith('can'):
        raise NotImplementedError("Tech '{}' does not start with 'can'".format(tech))

# energy_tanks = st.number_input("Energy tanks", min_value=0, max_value=14, step=1)
# reserve_tanks = st.number_input("Reserve tanks", min_value=0, max_value=4, step=1)
# missiles = st.number_input("Missiles", min_value=0, step=5)
# super_missiles = st.number_input("Super missiles", min_value=0, step=5)
# power_bombs = st.number_input("Power bombs", min_value=0, step=5)

# Update style to prevent text in options from being truncated:
st.markdown(
    """
    <style>
        .stMultiSelect [data-baseweb="tag"] {
            height: fit-content;
        }
        .stMultiSelect [data-baseweb="tag"] span[title] {
            white-space: normal; max-width: 100%; overflow-wrap: anywhere;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        .stSelectbox [data-baseweb="select"] div[aria-selected="true"] {
            white-space: normal; overflow-wrap: anywhere;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------- Tech ---------------------

tech_options = sorted([s[3:] for s in sm_json_data.tech_name_set])

if "all_tech" not in st.session_state:
    st.session_state.all_tech = True
    st.session_state.selected_tech = tech_options


def check_change_tech():
    if st.session_state.all_tech:
        st.session_state.selected_tech = tech_options
    else:
        st.session_state.selected_tech = []


def multi_change_tech():
    if len(st.session_state.selected_tech) == len(tech_options):
        st.session_state.all_tech = True
    else:
        st.session_state.all_tech = False


selected_tech = st.multiselect("Tech:", tech_options, key="selected_tech", on_change=multi_change_tech)
all_tech = st.checkbox("Select all", key='all_tech', on_change=check_change_tech)
shine_charge_tiles = st.number_input("Shine charge tiles", value=33, min_value=1, max_value=33)

# ---------- Flags -----------------------

flag_options = sorted(sm_json_data.flags_set)

if "all_flags" not in st.session_state:
    st.session_state.all_flags = True
    st.session_state.selected_flags = flag_options


def check_change_flags():
    if st.session_state.all_flags:
        st.session_state.selected_flags = flag_options
    else:
        st.session_state.selected_flags = []


def multi_change_flags():
    if len(st.session_state.selected_flags) == len(flag_options):
        st.session_state.all_flags = True
    else:
        st.session_state.all_flags = False


selected_flags = st.multiselect("Flags:", flag_options, key="selected_flags", on_change=multi_change_flags)
all_flags = st.checkbox("Select all", key='all_flags', on_change=check_change_flags)

# ---------- Region ----------------------
subregion_filename_dict = {'/'.join(filename.split('/')[2:])[:-5]: filename for filename in sm_json_data.region_json_dict.keys()}
subregion_name = st.selectbox("Region", sorted(list(subregion_filename_dict.keys())))
subregion_filename = subregion_filename_dict[subregion_name]

# ---------- Room ------------------------
room_options = sorted([room['name'] for room in sm_json_data.region_json_dict[subregion_filename]['rooms']])
room_name = st.selectbox("Room", room_options)
for room in sm_json_data.region_json_dict[subregion_filename]['rooms']:
    if room['name'] == room_name:
        break
else:
    raise RuntimeError("Room not found")

# ---------- Room image -----------------------
region = subregion_filename.split('/')[-2]
subregion_name = subregion_filename.split('/')[-1][:-5]
image_paths = [str(f) for f in pathlib.Path(f"{sm_json_data_path}/region/{region}").glob("roomDiagrams/*.png")]
prefix = f"{subregion_name}_{room['id']}"
for image_path in image_paths:
    filename = image_path.split('/')[-1]
    if filename.startswith(prefix):
        st.image(image_path)
        break
else:
    st.write("Room image not found")

# -------- Obstacles --------------------------
if 'obstacles' in room:
    obstacle_names = ["{}: {}".format(obstacle['id'], obstacle['name']) for obstacle in room['obstacles']]
    obstacle_ids = [obstacle['id'] for obstacle in room['obstacles']]
    obstacle_index_dict = {name: i for i, name in enumerate(obstacle_names)}
    cleared_obstacles_names = st.multiselect("Cleared obstacles", obstacle_names)
    cleared_obstacles_bitmask = sum(1 << obstacle_index_dict[name] for name in cleared_obstacles_names)
else:
    obstacle_ids = []
    obstacle_names = []
    cleared_obstacles_bitmask = 0

# -------- Node -------------------------------
node_names = ["{}: {} (vertex {})".format(node['id'], node['name'], sm_json_data.vertex_index_dict[(room['id'], node['id'], cleared_obstacles_bitmask)])
              for node in room['nodes']]
node_id_dict = {name: room['nodes'][i]['id'] for i, name in enumerate(node_names)}
from_node_name = st.selectbox("From node", node_names)
from_node_id = node_id_dict[from_node_name]

# ------------- Items ------------------

item_options = [item for item in sorted(sm_json_data.item_set)
                if item not in ("PowerBeam", "PowerSuit", "ETank", "ReserveTank")]

if "all_items" not in st.session_state:
    st.session_state.all_items = True
    st.session_state.selected_items = item_options


def check_change_items():
    if st.session_state.all_items:
        st.session_state.selected_items = item_options
    else:
        st.session_state.selected_items = []


def multi_change_items():
    if len(st.session_state.selected_items) == len(item_options):
        st.session_state.all_items = True
    else:
        st.session_state.all_items = False


selected_items = st.multiselect("Items:", item_options, key="selected_items", on_change=multi_change_items)
all_items = st.checkbox("Select all", key='all_items', on_change=check_change_items)

full_selected_items = selected_items + ["PowerBeam", "PowerSuit"]
# if missiles > 0:
#     full_selected_items += ["Missile"]
# if super_missiles > 0:
#     full_selected_items += ["Super"]
# if power_bombs > 0:
#     full_selected_items += ["PowerBomb"]

# -------- Logic ------------------------------
difficulty_config = DifficultyConfig(
    tech=set(['can' + tech for tech in selected_tech]),
    shine_charge_tiles=shine_charge_tiles,
    resource_multiplier=1.0,
    escape_time_multiplier=1.0,
    save_animals=True)
game_state = GameState(
    items=set(full_selected_items),
    flags=set(selected_flags),
    weapons=sm_json_data.get_weapons(set(full_selected_items)),
    num_energy_tanks=0,  # energy_tanks,
    num_reserves=0,  # reserve_tanks,
    max_energy=99,  # + 100 * (energy_tanks + reserve_tanks),
    max_missiles=0,  # missiles,
    max_super_missiles=0,  # super_missiles,
    max_power_bombs=0,  # power_bombs,
    current_energy=99,
    current_missiles=0,  # missiles,
    current_super_missiles=0,  # super_missiles,
    current_power_bombs=0,  # power_bombs,
    vertex_index=from_node_id)
from_vertex = sm_json_data.vertex_index_dict[(room['id'], from_node_id, cleared_obstacles_bitmask)]

link_by_to_node = defaultdict(lambda: [])
for link in sm_json_data.link_list:
    if link.from_index == from_vertex:
        to_room_id, to_node_id, to_obstacles_bitmask = sm_json_data.vertex_list[link.to_index]
        link_by_to_node[to_node_id].append(link)

for to_node_id in sorted(link_by_to_node.keys()):
    links = link_by_to_node[to_node_id]
    to_node_name = ['{}: {}'.format(to_node_id, node['name']) for node in room['nodes'] if node['id'] == to_node_id][0]
    with st.expander('To node {}'.format(to_node_name)):
        strat_name_list = []
        possible_list = []
        energy_list = []
        missiles_list = []
        supers_list = []
        pb_list = []
        cleared_obstacles_list = []
        for link in links:
            to_room_id, to_node_id, to_obstacles_bitmask = sm_json_data.vertex_list[link.to_index]
            to_obstacles_ids = [name for i, name in enumerate(obstacle_ids) if (1 << i) & to_obstacles_bitmask != 0]
            consumption = link.cond.get_consumption(game_state, difficulty_config)
            strat_name_list.append(link.strat_name)
            possible_list.append(consumption.possible)
            energy_list.append(consumption.energy)
            missiles_list.append(consumption.missiles)
            supers_list.append(consumption.super_missiles)
            pb_list.append(consumption.power_bombs)
            cleared_obstacles_list.append(to_obstacles_ids if consumption.possible else None)
        df = pd.DataFrame({
            "Strat": strat_name_list,
            "Possible": possible_list,
            "Energy": energy_list,
            "Missiles": missiles_list,
            "Supers": supers_list,
            "Power Bombs": pb_list,
            "Cleared Obstacles": cleared_obstacles_list}
        ).set_index("Strat")
        st.table(df)

st.json(room, expanded=False)