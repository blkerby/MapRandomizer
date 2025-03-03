import os

tilesets_path = "Mosaic/Projects/Base/Export/Tileset/SCE"
for path in sorted(os.listdir(tilesets_path)):
    tile_data = open(f'{tilesets_path}/{path}/16x16tiles.ttb', 'rb').read()
    bad_tiles = {}
    for i in range(0, min(len(tile_data), 6144), 2):
        x = tile_data[i] | (tile_data[i + 1] << 8)
        pal = (x >> 10) & 7
        c = x & 0x3ff
        if pal <= 1 and c < 640 and c > 32:
            if c not in bad_tiles:
                bad_tiles[c] = set()
            bad_tiles[c].add(i // 8)
    print("Tileset:", path)
    for c in sorted(bad_tiles.keys()):
        t = bad_tiles[c]
        print("{:03x}: {}".format(
            c, ', '.join('{:03x}'.format(x + 0x100) for x in sorted(t))))

