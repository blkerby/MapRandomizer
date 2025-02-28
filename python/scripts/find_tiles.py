import xml.etree.ElementTree as ET
import os


def get_int(subtree, selector):
    nodes = subtree.findall(selector)
    if len(nodes) > 0:
        return int(nodes[0].text, 16)
    else:
        return 0


selected_tilesets = [6]
selected_tiles = [0x335]
room_path = "Mosaic/Projects/Base/Export/Rooms"
for path in sorted(os.listdir(room_path)):
    room = ET.parse(f'{room_path}/{path}').getroot()
    states = room.findall("./States/State")
    for state_node in states:
        gfx_set = get_int(state_node, "./GFXset")
        if gfx_set not in selected_tilesets:
            continue
        level_data = state_node.findall("./LevelData")[0]
        width = int(level_data.attrib['Width'], 16)
        height = int(level_data.attrib['Height'], 16)
        state = {}
        state['level_data'] = [[None for _ in range(height)] for _ in range(width)]

        bts_nodes = state_node.findall("./LevelData/BTS/Screen")
        for bts_node in bts_nodes:
            x = int(bts_node.attrib['X'], 16)
            y = int(bts_node.attrib['Y'], 16)
            bts = [[int(b, 16)] for b in bts_node.text.split()]
            state['level_data'][x][y] = bts
        
        layer1_nodes = state_node.findall("./LevelData/Layer1/Screen")
        for layer1_node in layer1_nodes:
            x = int(layer1_node.attrib['X'], 16)
            y = int(layer1_node.attrib['Y'], 16)
            tiles = [int(t, 16) for t in layer1_node.text.split()]
            for i, tile in enumerate(tiles):
                state['level_data'][x][y][i].append(tiles[i])

        layer2_nodes = state_node.findall("./LevelData/Layer2/Screen")
        for layer2_node in layer2_nodes:
            x = int(layer2_node.attrib['X'], 16)
            y = int(layer2_node.attrib['Y'], 16)
            tiles = [int(t, 16) for t in layer2_node.text.split()]
            for i, tile in enumerate(tiles):
                state['level_data'][x][y][i].append(tiles[i])

        match_tiles = set()
        for y in range(height):
            for x in range(width):
                for i in range(256):
                    tile = state['level_data'][x][y][i]
                    bts = tile[0]
                    layer1 = tile[1]
                    layer2 = tile[2] if len(tile) >= 3 else None
                    
                    if layer1 & 0x3FF in selected_tiles:
                        match_tiles.add(layer1)
                    if layer2 is not None and layer2 & 0x3FF in selected_tiles:
                        match_tiles.add(layer2)
        
        if len(match_tiles) > 0:
            print("{}: gfx_set={:x}, tiles=[{}]".format(
                path, gfx_set, ', '.join('{:03x}'.format(t) for t in match_tiles)))
