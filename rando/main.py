from typing import List
from dataclasses import dataclass
from io import BytesIO
import numpy as np
import random
import graph_tool
import graph_tool.inference
import graph_tool.topology
from collections import defaultdict

# from rando.rooms import room_ptrs
from maze_builder.env import MazeBuilderEnv
from rando.sm_json_data import SMJsonData, GameState, Link, DifficultyConfig
from rando.rando import Randomizer
from logic.rooms.all_rooms import rooms
from maze_builder.display import MapDisplay
from maze_builder.types import Room
import json
import ips_util


# input_rom_path = '/home/kerby/Downloads/dash-rando-app-v9/DASH_v9_SM_8906529.sfc'
# input_rom_path = '/home/kerby/Downloads/Super Metroid Practice Hack-v2.3.3-emulator-ntsc.sfc'
input_rom_path = '/home/kerby/Downloads/Super Metroid (JU) [!].smc'
# map_name = '12-15-session-2021-12-10T06:00:58.163492-0'
# map_name = '01-16-session-2022-01-13T12:40:37.881929-1'



import torch
import logging
from maze_builder.types import reconstruct_room_data, Direction, DoorSubtype
import logic.rooms.all_rooms
import pickle

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])

torch.set_printoptions(linewidth=120, threshold=10000)
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

device = torch.device('cpu')
# session_name = '12-15-session-2021-12-10T06:00:58.163492'
# session_name = '01-16-session-2022-01-13T12:40:37.881929'
session_name = '04-16-session-2022-03-29T15:40:57.320430'
session = CPU_Unpickler(open('models/{}.pkl'.format(session_name), 'rb')).load()
ind = torch.nonzero(session.replay_buffer.episode_data.reward >= 343)
#

print(torch.sort(torch.sum(session.replay_buffer.episode_data.missing_connects.to(torch.float32), dim=0)))
print(torch.max(session.replay_buffer.episode_data.reward))

def get_map(ind_i):
    num_rooms = len(session.envs[0].rooms)
    action = session.replay_buffer.episode_data.action[ind[ind_i], :]
    step_indices = torch.tensor([num_rooms])
    room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)
    rooms = logic.rooms.all_rooms.rooms

    doors_dict = {}
    doors_cnt = {}
    door_pairs = []
    for i, room in enumerate(rooms):
        for door in room.door_ids:
            x = int(room_position_x[0, i]) + door.x
            if door.direction == Direction.RIGHT:
                x += 1
            y = int(room_position_y[0, i]) + door.y
            if door.direction == Direction.DOWN:
                y += 1
            vertical = door.direction in (Direction.DOWN, Direction.UP)
            key = (x, y, vertical)
            if key in doors_dict:
                a = doors_dict[key]
                b = door
                if a.direction in (Direction.LEFT, Direction.UP):
                    a, b = b, a
                if a.subtype == DoorSubtype.SAND:
                    door_pairs.append([[a.exit_ptr, a.entrance_ptr], [b.exit_ptr, b.entrance_ptr], False])
                else:
                    door_pairs.append([[a.exit_ptr, a.entrance_ptr], [b.exit_ptr, b.entrance_ptr], True])
                doors_cnt[key] += 1
            else:
                doors_dict[key] = door
                doors_cnt[key] = 1

    assert all(x == 2 for x in doors_cnt.values())
    map_name = '{}-{}'.format(session_name, ind_i)
    map = {
        'rooms': [[room_position_x[0, i].item(), room_position_y[0, i].item()]
                  for i in range(room_position_x.shape[1] - 1)],
        'doors': door_pairs
    }
    num_envs = 1
    episode_length = len(rooms)
    env = MazeBuilderEnv(rooms,
                         map_x=session.envs[0].map_x,
                         map_y=session.envs[0].map_y,
                         num_envs=num_envs,
                         device=device,
                         must_areas_be_connected=False)
    env.room_mask = room_mask
    env.room_position_x = room_position_x
    env.room_position_y = room_position_y
    # env.render(0)
    return map, map_name

# json.dump(map, open('maps/{}.json'.format(map_name), 'w'))

# map_name = '04-16-session-2022-03-29T15:40:57.320430-19'
get_map(77)



for ind_i in range(60, 100000):
    logging.info("ind_i={}".format(ind_i))
    map, map_name = get_map(ind_i=ind_i)
    map_path = 'maps/{}.json'.format(map_name)
    output_rom_path = 'roms/{}-c.sfc'.format(map_name)
    # map = json.load(open(map_path, 'r'))


    sm_json_data_path = "sm-json-data/"
    sm_json_data = SMJsonData(sm_json_data_path)
    tech = set(sm_json_data.tech_name_set)
    tech.remove('canHeatRun')
    # tech = set()
    difficulty = DifficultyConfig(tech=tech, shine_charge_tiles=33)

    randomizer = Randomizer(map['doors'], sm_json_data, difficulty)
    for i in range(0, 100):
        np.random.seed(i)
        random.seed(i)
        randomizer.randomize()
        # print(i, len(randomizer.item_placement_list))
        if len(randomizer.item_placement_list) >= 99:
            # i1 = randomizer.item_placement_list.index(544)
            # i2 = randomizer.item_placement_list.index(545)
            # print([randomizer.item_sequence[i1], randomizer.item_sequence[i2]])
            # if set([randomizer.item_sequence[i1], randomizer.item_sequence[i2]]) == set(["Gravity", "Missile"]):
            break
    else:
        continue
    break

print("Done with item randomization")

for room in rooms:
    room.populate()
xs_min = np.array([p[0] for p in map['rooms']])
ys_min = np.array([p[1] for p in map['rooms']])
xs_max = np.array([p[0] + rooms[i].width for i, p in enumerate(map['rooms'])])
ys_max = np.array([p[1] + rooms[i].height for i, p in enumerate(map['rooms'])])

room_graph = graph_tool.Graph(directed=True)
door_room_dict = {}
for i, room in enumerate(rooms):
    for door in room.door_ids:
        door_pair = (door.exit_ptr, door.entrance_ptr)
        door_room_dict[door_pair] = i
for conn in map['doors']:
    src_room_id = door_room_dict[tuple(conn[0])]
    dst_room_id = door_room_dict[tuple(conn[1])]
    room_graph.add_edge(src_room_id, dst_room_id)
    room_graph.add_edge(dst_room_id, src_room_id)

# Try to assign new areas to rooms in a way that makes areas as clustered as possible
best_entropy = float('inf')
best_state = None
num_areas = 6
for i in range(500, 2000):
    graph_tool.seed_rng(i)
    state = graph_tool.inference.minimize_blockmodel_dl(room_graph,
                                                        multilevel_mcmc_args={"B_min": num_areas, "B_max": num_areas})
    # for j in range(10):
    #     state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
    e = state.entropy()
    if e < best_entropy:
        u, block_id = np.unique(state.get_blocks().get_array(), return_inverse=True)
        assert len(u) == num_areas
        for j in range(num_areas):
            ind = np.where(block_id == j)
            x_range = np.max(xs_max[ind]) - np.min(xs_min[ind])
            y_range = np.max(ys_max[ind]) - np.min(ys_min[ind])
            if x_range > 60 or y_range > 30:
                break
        else:
            best_entropy = e
            best_state = state
    print(i, e, best_entropy)

assert best_state is not None
state = best_state
# state.draw()


display = MapDisplay(72, 72, 14)
_, area_arr = np.unique(state.get_blocks().get_array(), return_inverse=True)

# Ensure that Landing Site is in Crateria:
area_arr = (area_arr - area_arr[1] + num_areas) % num_areas

color_map = {
    0: (0x80, 0x80, 0x80),  # Crateria
    1: (0x80, 0xff, 0x80),  # Brinstar
    2: (0xff, 0x80, 0x80),  # Norfair
    3: (0xff, 0xff, 0x80),  # Wrecked ship
    4: (0x80, 0x80, 0xff),  # Maridia
    5: (0xc0, 0xc0, 0xc0),  # Tourian
}
colors = [color_map[i] for i in area_arr]
display.display(rooms, xs_min, ys_min, colors)



# print(randomizer.item_sequence[:5])
# print(randomizer.item_placement_list[:5])
# print(sm_json_data.node_list[697])

# randomizer.item_sequence.index("SpringBall")
# randomizer.item_placement_list[34]
# sm_json_data.node_list[679]


class Rom:
    def __init__(self, filename):
        file = open(filename, 'rb')
        self.bytes_io = BytesIO(file.read())
        self.byte_buf = self.bytes_io.getbuffer()

    def read_u8(self, pos):
        return self.byte_buf[pos]

    def read_u16(self, pos):
        return self.read_u8(pos) + (self.read_u8(pos + 1) << 8)

    def read_u24(self, pos):
        return self.read_u8(pos) + (self.read_u8(pos + 1) << 8) + (self.read_u8(pos + 2) << 16)

    def read_n(self, pos, n):
        return self.byte_buf[pos:(pos + n)]

    def write_u8(self, pos, value):
        self.byte_buf[pos] = value

    def write_u16(self, pos, value):
        self.byte_buf[pos] = value & 0xff
        self.byte_buf[pos + 1] = value >> 8

    def write_n(self, pos, n, values):
        self.byte_buf[pos:(pos + n)] = values

    def save(self, filename):
        file = open(filename, 'wb')
        file.write(self.bytes_io.getvalue())


area_map_ptrs = {
    0: 0x1A9000,  # Crateria
    1: 0x1A8000,  # Brinstar
    2: 0x1AA000,  # Norfair
    3: 0x1AB000,  # Wrecked ship
    4: 0x1AC000,  # Maridia
    5: 0x1AD000,  # Tourian
    6: 0x1AE000,  # Ceres
}

@dataclass
class Door:
    door_ptr: int
    dest_room_ptr: int
    bitflag: int
    direction: int
    door_cap_x: int
    door_cap_y: int
    screen_x: int
    screen_y: int
    dist_spawn: int

@dataclass
class RoomState:
    event_ptr: int   # u16
    event_value: int  # u8
    state_ptr: int   # u16
    level_data_ptr: int  # u24
    tile_set: int  # u8
    song_set: int  # u8
    play_index: int  # u8
    fx_ptr: int  # u16
    enemy_set_ptr: int  # u16
    enemy_gfx_ptr: int  # u16
    bg_scrolling: int  # u16
    room_scrolls_ptr: int  # u16
    unused_ptr: int  # u16
    main_asm_ptr: int  # u16
    plm_set_ptr: int  # u16
    bg_ptr: int  # u16
    setup_asm_ptr: int  # u16

class RomRoom:
    def __init__(self, rom: Rom, room: Room):
        room_ptr = room.rom_address
        self.room = room
        self.room_ptr = room_ptr
        self.area = rom.read_u8(room_ptr + 1)
        self.x = rom.read_u8(room_ptr + 2)
        self.y = rom.read_u8(room_ptr + 3)
        self.width = rom.read_u8(room_ptr + 4)
        self.height = rom.read_u8(room_ptr + 5)
        self.map_data = self.load_map_data(rom)
        self.doors = self.load_doors(rom)
        # self.load_states(rom)

    def load_map_data(self, rom):
        map_row_list = []
        for y in range(self.y, self.y + self.room.height):
            map_row = []
            for x in range(self.x, self.x + self.room.width):
                cell = rom.read_u16(self.xy_to_map_ptr(x, y))
                # if self.room.map[y - self.y][x - self.x] == 0:
                #     cell = 0x1F  # Empty tile
                map_row.append(cell)
            map_row_list.append(map_row)
        return map_row_list


    def load_single_state(self, rom, event_ptr, event_value, state_ptr):
        return RoomState(
            event_ptr=event_ptr,
            event_value=event_value,
            state_ptr=state_ptr,
            level_data_ptr=rom.read_u24(state_ptr),
            tile_set=rom.read_u8(state_ptr + 3),
            song_set=rom.read_u8(state_ptr + 4),
            play_index=rom.read_u8(state_ptr + 5),
            fx_ptr=rom.read_u16(state_ptr + 6),
            enemy_set_ptr=rom.read_u16(state_ptr + 8),
            enemy_gfx_ptr=rom.read_u16(state_ptr + 10),
            bg_scrolling=rom.read_u16(state_ptr + 12),
            room_scrolls_ptr=rom.read_u16(state_ptr + 14),
            unused_ptr=rom.read_u16(state_ptr + 16),
            main_asm_ptr=rom.read_u16(state_ptr + 18),
            plm_set_ptr=rom.read_u16(state_ptr + 20),
            bg_ptr=rom.read_u16(state_ptr + 22),
            setup_asm_ptr=rom.read_u16(state_ptr + 24),
        )

    def load_states(self, rom) -> List[RoomState]:
        ss = []
        for i in range(400):
            ss.append("{:02x} ".format(rom.read_u8(self.room_ptr + i)))
            if i % 16 == 0:
                ss.append("\n")
        # print(''.join(ss))
        pos = 11
        states = []
        while True:
            ptr = rom.read_u16(self.room_ptr + pos)
            # print("{:x}".format(ptr))
            if ptr == 0xE5E6:
                # This is the standard state, which is the last one
                event_value = 0  # Dummy value
                state_ptr = self.room_ptr + pos + 2
                states.append(self.load_single_state(rom, ptr, event_value, state_ptr))
                break
            elif ptr in (0xE612, 0xE629):
                # This is an event state
                event_value = rom.read_u8(self.room_ptr + pos + 2)
                state_ptr = 0x70000 + rom.read_u16(self.room_ptr + pos + 3)
                states.append(self.load_single_state(rom, ptr, event_value, state_ptr))
                pos += 5
            else:
                event_value = 0  # Dummy value
                state_ptr = 0x70000 + rom.read_u16(self.room_ptr + pos + 2)
                states.append(self.load_single_state(rom, ptr, event_value, state_ptr))
                pos += 4
        return states

    def load_doors(self, rom):
        self.doors = []
        door_out_ptr = 0x70000 + rom.read_u16(self.room_ptr + 9)
        while True:
            door_ptr = 0x10000 + rom.read_u16(door_out_ptr)
            if door_ptr < 0x18000:
                break
            door_out_ptr += 2

            dest_room_ptr = rom.read_u16(door_ptr)
            bitflag = rom.read_u8(door_ptr + 2)
            direction = rom.read_u8(door_ptr + 3)
            door_cap_x = rom.read_u8(door_ptr + 4)
            door_cap_y = rom.read_u8(door_ptr + 5)
            screen_x = rom.read_u8(door_ptr + 6)
            screen_y = rom.read_u8(door_ptr + 7)
            dist_spawn = rom.read_u16(door_ptr + 8)
            door_asm = rom.read_u16(door_ptr + 10)
            self.doors.append(Door(
                door_ptr=door_ptr,
                dest_room_ptr=dest_room_ptr,
                # horizontal=direction in [2, 3, 6, 7],
                bitflag=bitflag,
                direction=direction,
                screen_x=screen_x,
                screen_y=screen_y,
                door_cap_x=door_cap_x,
                door_cap_y=door_cap_y,
                dist_spawn=dist_spawn,
            ))
            # print(f'{dest_room_ptr:x} {bitflag} {direction} {door_cap_x} {door_cap_y} {screen_x} {screen_y} {dist_spawn:x} {door_asm:x}')

    def save_doors(self, rom):
        for door in self.doors:
            door_ptr = door.door_ptr
            rom.write_u16(door_ptr, door.dest_room_ptr)
            rom.write_u8(door_ptr + 2, door.bitflag)
            rom.write_u8(door_ptr + 3, door.direction)
            rom.write_u8(door_ptr + 4, door.door_cap_x)
            rom.write_u8(door_ptr + 5, door.door_cap_y)
            rom.write_u8(door_ptr + 6, door.screen_x)
            rom.write_u8(door_ptr + 7, door.screen_y)
            rom.write_u16(door_ptr + 8, door.dist_spawn)

    def xy_to_map_ptr(self, x, y):
        base_ptr = area_map_ptrs[self.area]
        y1 = y + 1
        if x < 32:
            offset = (y1 * 32 + x) * 2
        else:
            offset = ((y1 + 32) * 32 + x - 32) * 2
        return base_ptr + offset

    def write_map_data(self, rom):
        # rom.write_u8(self.room_ptr + 1, self.area)
        rom.write_u8(self.room_ptr + 2, self.x)
        rom.write_u8(self.room_ptr + 3, self.y)

        for y in range(self.room.height):
            for x in range(self.room.width):
                ptr = self.xy_to_map_ptr(x + self.x, y + self.y)
                if self.room.map[y][x] == 1:
                    rom.write_u16(ptr, self.map_data[y][x])


orig_rom = Rom(input_rom_path)
rom = Rom(input_rom_path)

# Change Aqueduct map y position, to include the toilet (for the purposes of the map)
old_y = orig_rom.read_u8(0x7D5A7 + 3)
orig_rom.write_u8(0x7D5A7 + 3, old_y - 4)

# # Change door asm for entering mother brain room
orig_rom.write_u16(0x1AAC8 + 10, 0xEB00)
# rom.write_u16(0x1956A + 10, 0xEB00)


# Area data: --------------------------------
area_index_dict = defaultdict(lambda: {})
for i, room in enumerate(rooms):
    orig_room_area = orig_rom.read_u8(room.rom_address + 1)
    room_index = orig_rom.read_u8(room.rom_address)
    assert room_index not in area_index_dict[orig_room_area]
    area_index_dict[orig_room_area][room_index] = area_arr[i]
# Handle twin rooms
aqueduct_room_i = [i for i, room in enumerate(rooms) if room.name == 'Aqueduct'][0]
area_index_dict[4][0x18] = area_arr[aqueduct_room_i]  # Set Toilet to same area as Aqueduct
pants_room_i = [i for i, room in enumerate(rooms) if room.name == 'Pants Room'][0]
area_index_dict[4][0x25] = area_arr[pants_room_i]  # Set East Pants Room to same area as Pants Room
west_ocean_room_i = [i for i, room in enumerate(rooms) if room.name == 'West Ocean'][0]
area_index_dict[0][0x11] = area_arr[west_ocean_room_i]  # Set Homing Geemer Room to same area as West Ocean
# Write area data
area_sizes = [max(area_index_dict[i].keys()) + 1 for i in range(num_areas)]
cumul_area_sizes = [0] + list(np.cumsum(area_sizes))
area_data_base_ptr = 0x7E99B  # LoRom $8F:E99B
area_data_ptrs = [area_data_base_ptr + num_areas * 2 + cumul_area_sizes[i] for i in range(num_areas)]
assert area_data_ptrs[-1] <= 0x7EB00
for i in range(num_areas):
    rom.write_u16(area_data_base_ptr + 2 * i, (area_data_ptrs[i] & 0x7FFF) + 0x8000)
    for room_index, new_area in area_index_dict[i].items():
        rom.write_u8(area_data_ptrs[i] + room_index, new_area)

print("{:x}".format(area_data_ptrs[-1] + area_sizes[-1]))


# Write map data:
# first clear existing maps
for area_id, area_ptr in area_map_ptrs.items():
    for i in range(64 * 32):
        # if area_id == 0:
        #     rom.write_u16(area_ptr + i * 2, 0x0C1F)
        # else:
            rom.write_u16(area_ptr + i * 2, 0x001F)

area_start_x = []
area_start_y = []
for i in range(num_areas):
    ind = np.where(area_arr == i)
    area_start_x.append(np.min(xs_min[ind]) - 2)
    area_start_y.append(np.min(ys_min[ind]) - 1)

for i, room in enumerate(rooms):
    rom_room = RomRoom(orig_rom, room)
    area = area_arr[i]
    rom_room.area = area
    rom_room.x = xs_min[i] - area_start_x[area]
    rom_room.y = ys_min[i] - area_start_y[area]
    rom_room.write_map_data(rom)
    if room.name == 'Aqueduct':
        # Patch map tile in Aqueduct to replace Botwoon Hallway with tube/elevator tile
        cell = rom.read_u16(rom_room.xy_to_map_ptr(rom_room.x + 2, rom_room.y + 2))
        rom.write_u16(rom_room.xy_to_map_ptr(rom_room.x + 2, rom_room.y + 3), cell)


def write_door_data(ptr, data):
    if ptr in (0x1A600, 0x1A60C):
        # Avoid overwriting the door ASM leaving the toilet room. Otherwise Samus will be stuck,
        # unable to be controlled. This is only quick hack because by not applying the door ASM for
        # the next room, this can mess up camera scrolls and other things. (At some point,
        # maybe figure out how we can patch both ASMs together.)
        rom.write_n(ptr, 10, data[:10])
    else:
        rom.write_n(ptr, 12, data)
    bitflag = data[2] | 0x40
    rom.write_u8(ptr + 2, bitflag)
    # print("{:x}".format(bitflag))

def write_door_connection(a, b):
    a_exit_ptr, a_entrance_ptr = a
    b_exit_ptr, b_entrance_ptr = b
    if a_entrance_ptr is not None and b_exit_ptr is not None:
        # print('{:x},{:x}'.format(a_entrance_ptr, b_exit_ptr))
        a_entrance_data = orig_rom.read_n(a_entrance_ptr, 12)
        write_door_data(b_exit_ptr, a_entrance_data)
        # rom.write_n(b_exit_ptr, 12, a_entrance_data)
    if b_entrance_ptr is not None and a_exit_ptr is not None:
        b_entrance_data = orig_rom.read_n(b_entrance_ptr, 12)
        write_door_data(a_exit_ptr, b_entrance_data)
        # rom.write_n(a_exit_ptr, 12, b_entrance_data)
        # print('{:x} {:x}'.format(b_entrance_ptr, a_exit_ptr))

for (a, b, _) in list(map['doors']):
    write_door_connection(a, b)


save_station_ptrs = [
    0x44C5,
    0x44D3,
    0x45CF,
    0x45DD,
    0x45EB,
    0x45F9,
    0x4607,
    0x46D9,
    0x46E7,
    0x46F5,
    0x4703,
    0x4711,
    0x471F,
    0x481B,
    0x4917,
    0x4925,
    0x4933,
    0x4941,
    0x4A2F,
    0x4A3D,
]

area_save_ptrs = [0x44C5, 0x45CF, 0x46D9, 0x481B, 0x4917, 0x4A2F]

orig_door_dict = {}
for room in rooms:
    for door in room.door_ids:
        orig_door_dict[door.exit_ptr] = door.entrance_ptr
        # if door.exit_ptr is not None:
        #     door_asm = orig_rom.read_u16(door.exit_ptr + 10)
        #     if door_asm != 0:
        #         print("{:x}".format(door_asm))



door_dict = {}
for (a, b, _) in map['doors']:
    a_exit_ptr, a_entrance_ptr = a
    b_exit_ptr, b_entrance_ptr = b
    if a_exit_ptr is not None and b_exit_ptr is not None:
        door_dict[a_exit_ptr] = b_exit_ptr
        door_dict[b_exit_ptr] = a_exit_ptr


# Fix save stations
for ptr in save_station_ptrs:
    orig_entrance_door_ptr = orig_rom.read_u16(ptr + 2) + 0x10000
    exit_door_ptr = orig_door_dict[orig_entrance_door_ptr]
    entrance_door_ptr = door_dict[exit_door_ptr]
    rom.write_u16(ptr + 2, entrance_door_ptr & 0xffff)
#
# # Fix save stations
# room_ptr_to_idx = {room.rom_address: i for i, room in enumerate(rooms)}
# area_save_idx = {x: 0 for x in range(6)}
# area_save_idx[0] = 1  # Start Crateria index at 1 since we keep ship save station as is.
# for ptr in save_station_ptrs:
#     room_ptr = orig_rom.read_u16(ptr) + 0x70000
#     if room_ptr != 0x791F8:  # The ship has no Save Station PLM for us to update (and we don't need to since we keep the ship in Crateria)
#         room_obj = Room(orig_rom, room_ptr)
#         states = room_obj.load_states(orig_rom)
#         plm_ptr = states[0].plm_set_ptr + 0x70000
#         plm_type = orig_rom.read_u16(plm_ptr)
#         assert plm_type == 0xB76F  # Check that the first PLM is a save station
#
#         area = cs[room_ptr_to_idx[room_ptr]]
#         idx = area_save_idx[area]
#         rom.write_u16(plm_ptr + 4, area_save_idx[area])
#         area_save_idx[area] += 1
#
#         orig_save_station_bytes = orig_rom.read_n(ptr, 14)
#         dst_ptr = area_save_ptrs[area] + 14 * idx
#         rom.write_n(dst_ptr, 14, orig_save_station_bytes)
#     else:
#         area = 0
#         dst_ptr = ptr
#
#     orig_entrance_door_ptr = rom.read_u16(dst_ptr + 2) + 0x10000
#     exit_door_ptr = orig_door_dict[orig_entrance_door_ptr]
#     entrance_door_ptr = door_dict[exit_door_ptr] & 0xffff
#     rom.write_u16(dst_ptr + 2, entrance_door_ptr & 0xffff)

# item_dict = {}
for room_obj in rooms:
    if room_obj.name == 'Pit Room':
        # Leave grey doors in Pit Room intact, so that there is a way to trigger Zebes becoming awake.
        continue
    room = RomRoom(orig_rom, room_obj)
    states = room.load_states(rom)
    for state in states:
        ptr = state.plm_set_ptr + 0x70000
        while True:
            plm_type = orig_rom.read_u16(ptr)
            if plm_type == 0:
                break
            # if plm_type in (0xC842, 0xC848, 0xC84E, 0xC854):
            #     print('{}: {:04x} {:04x}'.format(room_obj.name, rom.read_u16(ptr + 2), rom.read_u16(ptr + 4)))
            #     rom.write_u8(ptr + 5, 0x0)  # main boss dead
            #     # rom.write_u8(ptr + 5, 0x0C)  # enemies dead

            # # Collect item ids
            # if (plm_type >> 8) in (0xEE, 0xEF):
            #     item_type_index = rando.conditions.get_plm_type_item_index(plm_type)
            #     print("{}: {}".format(room_obj.name, rando.conditions.item_list[item_type_index]))
            #     item_dict[ptr] = plm_type
            #     # print("{:x} {:x} {:x}".format(ptr, plm_type, item_id))

            # Turn non-blue doors blue
            if plm_type in (0xC88A, 0xC842, 0xC85A, 0xC872):  # right grey/yellow/green door
                # print('{}: {:x} {:x} {:x}'.format(room_obj.name, rom.read_u16(ptr), rom.read_u16(ptr + 2), rom.read_u16(ptr + 4)))
                # rom.write_u16(ptr, 0xC88A)  # right pink door
                rom.write_u16(ptr, 0xB63B)  # right continuation arrow (should have no effect, giving a blue door)
                rom.write_u16(ptr + 2, 0)  # position = (0, 0)
            elif plm_type in (0xC890, 0xC848, 0xC860, 0xC878):  # left grey/yellow/green door
                # rom.write_u16(ptr, 0xC890)  # left pink door
                rom.write_u16(ptr, 0xB63B)  # right continuation arrow (should have no effect, giving a blue door)
                rom.write_u16(ptr + 2, 0)  # position = (0, 0)
            elif plm_type in (0xC896, 0xC84E, 0xC866, 0xC87E):  # down grey/yellow/green door
                # rom.write_u16(ptr, 0xC896)  # down pink door
                rom.write_u16(ptr, 0xB63B)  # right continuation arrow (should have no effect, giving a blue door)
                rom.write_u16(ptr + 2, 0)  # position = (0, 0)
            elif plm_type in (0xC89C, 0xC854, 0xC86C, 0xC884):  # up grey/yellow/green door
                # rom.write_u16(ptr, 0xC89C)  # up pink door
                rom.write_u16(ptr, 0xB63B)  # right continuation arrow (should have no effect, giving a blue door)
                rom.write_u16(ptr + 2, 0)  # position = (0, 0)
            ptr += 6

def item_to_plm_type(item_name, orig_plm_type):
    item_list = [
        "ETank",
        "Missile",
        "Super",
        "PowerBomb",
        "Bombs",
        "Charge",
        "Ice",
        "HiJump",
        "SpeedBooster",
        "Wave",
        "Spazer",
        "SpringBall",
        "Varia",
        "Gravity",
        "XRayScope",
        "Plasma",
        "Grapple",
        "SpaceJump",
        "ScrewAttack",
        "Morph",
        "ReserveTank",
    ]
    i = item_list.index(item_name)
    old_i = ((orig_plm_type - 0xEED7) // 4) % 21
    return orig_plm_type + (i - old_i) * 4

# Place items
for i in range(len(randomizer.item_placement_list)):
    index = randomizer.item_placement_list[i]
    item_name = randomizer.item_sequence[i]
    ptr = sm_json_data.node_ptr_list[index]
    orig_plm_type = orig_rom.read_u16(ptr)
    plm_type = item_to_plm_type(item_name, orig_plm_type)
    rom.write_u16(ptr, plm_type)

# Make whole map revealed (after getting map station), i.e. no more "secret rooms" that don't show up in map.
for i in range(0x11727, 0x11D27):
    rom.write_u8(i, 0xFF)


# print(randomizer.item_sequence[:5])
# print(randomizer.item_placement_list[:5])
# sm_json_data.node_list[641]


# # Randomize items
# item_list = list(item_dict.values())
# item_perm = np.random.permutation(len(item_dict.values()))
# for i, ptr in enumerate(item_dict.keys()):
#     item = item_list[item_perm[i]]
#     rom.write_u16(ptr, item)

# rom.write_u16(0x78000, 0xC82A)
# rom.write_u8(0x78002, 40)
# rom.write_u8(0x78003, 68)
# rom.write_u16(0x78004, 0x8000)


# ---- Fix twin room map x & y:
# Aqueduct:
old_aqueduct_x = rom.read_u8(0x7D5A7 + 2)
old_aqueduct_y = rom.read_u8(0x7D5A7 + 3)
rom.write_u8(0x7D5A7 + 3, old_aqueduct_y + 4)
# Toilet:
rom.write_u8(0x7D408 + 2, old_aqueduct_x + 2)
rom.write_u8(0x7D408 + 3, old_aqueduct_y)
# East Pants Room:
pants_room_x = rom.read_u8(0x7D646 + 2)
pants_room_y = rom.read_u8(0x7D646 + 3)
rom.write_u8(0x7D69A + 2, pants_room_x + 1)
rom.write_u8(0x7D69A + 3, pants_room_y + 1)
# Homing Geemer Room:
west_ocean_x = rom.read_u8(0x793FE + 2)
west_ocean_y = rom.read_u8(0x793FE + 3)
rom.write_u8(0x7968F + 2, west_ocean_x + 5)
rom.write_u8(0x7968F + 3, west_ocean_y + 2)

# Apply patches
patches = [
    'vanilla_bugfixes',
    'new_game',
    # 'new_game_extra',
    'music',
    'crateria_sky',
    'everest_tube',
    'sandfalls',
    'escape_room_1',
    'saveload',
    'map_area',
    'mb_barrier',
    'mb_barrier_clear',  # Seems to incompatible with fast_doors due to race condition with how level data is loaded (which fast_doors speeds up)?
    # 'fast_doors',
    'elevators_speed',
    'boss_exit',
    'itemsounds',
    'progressive_suits',
    'disable_map_icons',
    'escape',
]
for patch_name in patches:
    patch = ips_util.Patch.load('patches/ips/{}.ips'.format(patch_name))
    rom.byte_buf = patch.apply(rom.byte_buf)



# rom.write_u16(0x79213 + 24, 0xEB00)
# rom.write_u16(0x7922D + 24, 0xEB00)
# rom.write_u16(0x79247 + 24, 0xEB00)
# rom.write_u16(0x79247 + 24, 0xEB00)
# rom.write_u16(0x79261 + 24, 0xEB00)

# Connect bottom left landing site door to mother brain room, for testing
# mb_door_bytes = orig_rom.read_n(0X1AAC8, 12)
# rom.write_n(0x18916, 12, mb_door_bytes)

# Change setup asm for Mother Brain room
rom.write_u16(0x7DD6E + 24, 0xEB00)



# Change door exit asm for boss rooms (TODO: do this better, in case entrance asm is needed in next room)
boss_exit_asm = 0xF7F0
# Kraid:
rom.write_u16(0x191CE + 10, boss_exit_asm)
rom.write_u16(0x191DA + 10, boss_exit_asm)
# Draygon:
rom.write_u16(0x1A978 + 10, boss_exit_asm)
rom.write_u16(0x1A96C + 10, boss_exit_asm)


# print("{:x}".format(rom.read_u16(0x786DE)))


with open(output_rom_path, 'wb') as out_file:
    out_file.write(rom.byte_buf)

print("Wrote to {}".format(output_rom_path))