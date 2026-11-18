import logging
import argparse
import json
import fastavro
import pathlib
import sys
import torch
from logic.rooms.all_rooms import rooms

sys.path.append('map-gen/python')

toilet_idx = next(i for i, room in enumerate(rooms) if room.name == 'Toilet')
aqueduct_idx = next(i for i, room in enumerate(rooms) if room.name == 'Aqueduct')
landing_site_idx = next(i for i, room in enumerate(rooms) if room.name == 'Landing Site')
door_map = {}
door_room_map = {}
for room_i, room in enumerate(rooms):
    for door_id, door in enumerate(room.door_ids):
        door_map[(door.exit_ptr, door.entrance_ptr)] = (room.room_id, door_id)
        door_room_map[(door.exit_ptr, door.entrance_ptr)] = room_i


def convert_map(input_dict):
    if len(input_dict['rooms']) == len(rooms) - 1:
        # The old Aqueduct origin included four rows above the actual room.
        x, y = input_dict['rooms'][aqueduct_idx]
        input_dict['rooms'][aqueduct_idx] = [x, y + 4]
        input_dict['rooms'].insert(toilet_idx, [x + 2, y])
        for field in ['area', 'subarea']:
            input_dict[field].insert(toilet_idx, input_dict[field][aqueduct_idx])
        input_dict['toilet_intersections'] = [aqueduct_idx]

    if 'subsubarea' not in input_dict:
        from subarea import split_assignment_by_balanced_centers

        # Contract Toilet into its crossing room so they receive the same subdivision.
        room_indices = [i for i in range(len(rooms)) if i != toilet_idx]
        room_index_map = {room_i: i for i, room_i in enumerate(room_indices)}
        intersection = input_dict['toilet_intersections'][0]
        room_index_map[toilet_idx] = room_index_map[intersection]
        adjacency = torch.eye(len(room_indices), dtype=torch.bool)
        for src, dst, _ in input_dict['doors']:
            src_i = room_index_map[door_room_map[tuple(src)]]
            dst_i = room_index_map[door_room_map[tuple(dst)]]
            adjacency[src_i, dst_i] = True
            adjacency[dst_i, src_i] = True
        parents = tuple(torch.tensor(input_dict[field])[None, room_indices]
                        for field in ['area', 'subarea'])
        subsubareas = split_assignment_by_balanced_centers(
            adjacency[None], parents, (6, 2))[0]
        subsubareas = (subsubareas - subsubareas[room_index_map[landing_site_idx]]) % 2
        input_dict['subsubarea'] = subsubareas.tolist()
        input_dict['subsubarea'].insert(toilet_idx, subsubareas[room_index_map[intersection]].item())

    return {
        "room_id": [rooms[i].room_id for i in range(len(input_dict['rooms']))],
        "room_x": [room[0] for room in input_dict["rooms"]],
        "room_y": [room[1] for room in input_dict["rooms"]],
        "room_area": input_dict["area"],
        "room_subarea": input_dict["subarea"],
        "room_subsubarea": input_dict["subsubarea"],
        "conn_from_room_id": [door_map[tuple(d[0])][0] for d in input_dict["doors"]],
        "conn_from_door_id": [door_map[tuple(d[0])][1] for d in input_dict["doors"]],
        "conn_to_room_id": [door_map[tuple(d[1])][0] for d in input_dict["doors"]],
        "conn_to_door_id": [door_map[tuple(d[1])][1] for d in input_dict["doors"]],
        "conn_bidirectional": [d[2] for d in input_dict["doors"]],
    }


def main():
    parser = argparse.ArgumentParser(
        'convert_map_avro',
        'Convert map JSON files into Avro files')
    parser.add_argument('src_path', type=pathlib.Path)
    parser.add_argument('dst_path', type=pathlib.Path)
    parser.add_argument('maps_per_file', type=int, help='Maximum maps per Avro file')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO,
                        handlers=[logging.FileHandler('convert_map_avro.log'),
                                  logging.StreamHandler()])
    avro_map_schema = fastavro.parse_schema(json.load(open("rust/data/schema/map.avsc", "r")))
    args.dst_path.mkdir(parents=True, exist_ok=True)
    file_list = sorted(x for x in args.src_path.iterdir()
                       if x.suffix == '.json' and x.name != 'manifest.json')
    output_file_list = []
    maps_per_file = []
    file_count = (len(file_list) + args.maps_per_file - 1) // args.maps_per_file
    for file_i in range(file_count):
        # Batches are selected uniformly, so keep their sizes as equal as possible.
        start = file_i * len(file_list) // file_count
        end = (file_i + 1) * len(file_list) // file_count
        output_filename = f"maps-{len(output_file_list)}.avro"
        output_path = args.dst_path / output_filename
        records = []
        for input_path in file_list[start:end]:
            with input_path.open() as source:
                records.append(convert_map(json.load(source)))

        logging.info(f"Writing {output_path}")
        with output_path.open('wb') as output_file:
            fastavro.writer(output_file, avro_map_schema, records, codec='deflate')
        output_file_list.append(output_filename)
        maps_per_file.append(len(records))

    manifest = {"maps_per_file": maps_per_file, "files": output_file_list}
    with (args.dst_path / 'manifest.json').open('w') as output_file:
        json.dump(manifest, output_file)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
