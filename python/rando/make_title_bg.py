import numpy as np

def encode_graphics(image):
    image0 = image
    image = np.zeros([256, 1024, 3])
    image[:224, :256, :] = image0
    tile_size_x = 8
    tile_size_y = 8
    num_tiles_y = image.shape[0] // tile_size_y
    num_tiles_x = image.shape[1] // tile_size_x
    image_tiled = image.reshape([num_tiles_y, tile_size_y, num_tiles_x, tile_size_x, 3])
    tiles = np.transpose(image_tiled, axes=[0, 2, 1, 3, 4]).reshape([num_tiles_y * num_tiles_x, tile_size_y * tile_size_x, 3])
    tilemap = np.zeros([num_tiles_y * num_tiles_x], dtype=np.uint8)

    tiles_list = []
    tiles_dict = {}
    num_distinct_tiles = 0

    # Add the Map Station as the first 4 tiles:
    for i in [20 * 128 + 15, 20 * 128 + 16, 21 * 128 + 15, 21 * 128 + 16]:
        data = tuple(tiles[i, :].reshape([-1]))
        # assert data not in tiles_dict
        tiles_dict[data] = num_distinct_tiles
        tiles_list.append(data)
        num_distinct_tiles += 1

    for i in range(tiles.shape[0]):
        data = tuple(tiles[i, :].reshape([-1]))
        if data in tiles_dict:
            tilemap[i] = tiles_dict[data]
        else:
            tilemap[i] = num_distinct_tiles
            tiles_dict[data] = num_distinct_tiles
            tiles_list.append(data)
            num_distinct_tiles += 1
    print("Distinct tiles: {}".format(num_distinct_tiles))
    if num_distinct_tiles > 256:
        raise RuntimeError("Too many tiles")

    gfx = np.zeros([256, tile_size_y * tile_size_x], dtype=np.uint8)
    color_list = []
    color_dict = {}
    num_distinct_colors = 0
    for i, tile in enumerate(tiles_list):
        for j in range(len(tile) // 3):
            color = tuple(min((c + 4) // 8 * 8, 255) for c in tile[(j * 3):((j + 1) * 3)])
            if color in color_dict:
                gfx[i, j] = color_dict[color]
            else:
                gfx[i, j] = num_distinct_colors
                color_dict[color] = num_distinct_colors
                color_list.append(color)
                num_distinct_colors += 1
    print("Distinct colors: {}".format(num_distinct_colors))
    if num_distinct_colors > 256:
        raise RuntimeError("Too many colors")

    pal_arr = np.array(color_list, dtype=np.uint16) >> 3
    pal = pal_arr[:, 0] + (pal_arr[:, 1] << 5) + (pal_arr[:, 2] << 10)
    return pal, gfx, tilemap
