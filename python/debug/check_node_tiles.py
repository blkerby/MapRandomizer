from logic.rooms.all_rooms import rooms

for room in rooms:
    try:
        all_coords_set = set()
        for y in range(len(room.map)):
            for x in range(len(room.map[0])):
                if room.map[y][x] == 1:
                    all_coords_set.add((x, y))

        covered_coords_set = set()
        tile_list = list(room.node_tiles.items())
        if room.twin_node_tiles is not None:
            tile_list = tile_list + list(room.twin_node_tiles.items())
        for k, v in tile_list:
            for (x, y) in v:
                if (x, y) not in all_coords_set:
                    print("Room='{}', Node={}, coords=({}, {}): not valid".format(room.name, k, x, y))
                covered_coords_set.add((x, y))

        for (x, y) in all_coords_set:
            if (x, y) not in covered_coords_set:
                print("Room='{}', coords=({}, {}): not covered".format(room.name, x, y))
    except Exception as e:
        print("Room='{}': {}".format(room.name, e))

