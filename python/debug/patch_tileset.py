from rando.rom import Rom, snes2pc, RomRoom, pc2snes
from rando.compress import compress
from debug.decompress import decompress
import numpy as np
import png
from logic.rooms.all_rooms import rooms
from collections import defaultdict
import ips_util
from io import BytesIO
import flask


def write_palette(rom, addr, pal_arr):
    pal_arr = pal_arr.astype(np.int16)
    pal_r = pal_arr[:, 0] // 8
    pal_g = pal_arr[:, 1] // 8
    pal_b = pal_arr[:, 2] // 8
    pal = pal_r + (pal_g << 5) + (pal_b << 10)
    pal_bytes = pal.tobytes()
    compressed_bytes = compress(pal_bytes)
    rom.write_n(addr, len(compressed_bytes), compressed_bytes)
    return len(compressed_bytes)


input_rom_path = '/home/kerby/Downloads/Super Metroid (JU) [!].smc'
# input_rom_path = '/home/kerby/Downloads/Super Metroid Practice Hack-v2.5.1-sd2snes-ntsc.sfc'
# input_tilesets_path = '/home/kerby/SM_Tilesets_patched/'
# output_rom_path = '/home/kerby/Downloads/tiletest.smc'

def create_patched_rom(tileset_idxs, file_bytes_list):
    rom = Rom(open(input_rom_path, 'rb'))

    # free_space = 0xB88000
    free_space = 0xE18000

    for tileset_i, file_bytes in zip(tileset_idxs, file_bytes_list):
        reader = png.Reader(BytesIO(file_bytes))
        reader.preamble()
        palette = np.array(reader.palette())
        rom.write_u24(snes2pc(0x8FE6A2 + tileset_i * 9 + 6), free_space)
        free_space += write_palette(rom, snes2pc(free_space), palette)

    # rom.write_n(snes2pc(0x8DC4C8), 3, bytes(3 * [0xEA]))  # disable palette FX
    rom.write_u8(snes2pc(0x8DC527), 0x6B)  # RTL  (skip palette FX handler)

    patches = [
        'new_game_extra',
    ]
    for patch_name in patches:
        patch = ips_util.Patch.load('patches/ips/{}.ips'.format(patch_name))
        rom.bytes_io = BytesIO(patch.apply(rom.bytes_io.getvalue()))

    return rom
# for addr in norfair_glow:
#     color = rom.read_u16(addr)
#     r = color & 0x1F
#     g = (color >> 5) & 0x1F
#     b = (color >> 10) & 0x1F
#
#     x = (r + g + b) // 3
#     new_color = x + (x << 5) + (x << 10)
#     rom.write_u16(addr, new_color)
#     print("{:x}: {} {} {}".format(addr, r, g, b))

# rom.save(output_rom_path)


app = flask.Flask(__name__)

@app.route("/")
def home():
    return '''
<html>
<head>
   <title>Tileset tester</title>
</head>
<body>
    <form method="POST" enctype="multipart/form-data" action="/patch">
      <input type="file" name="pngFiles" multiple><br><br>
      <br>
      <input type="submit" value="Submit">
    </form>
</body>
</html>
'''

@app.route("/patch", methods=['POST'])
def patch():
    png_files = flask.request.files.getlist('pngFiles')
    tileset_idxs = []
    file_bytes_list = []
    for file in png_files:
        tileset_idxs.append(int(file.filename.split('.')[0]))
        file_bytes_list.append(file.stream.read())
    rom = create_patched_rom(tileset_idxs, file_bytes_list)
    rom.bytes_io.seek(0)
    return flask.send_file(rom.bytes_io, download_name='tileset_test.sfc')


app.run(host='0.0.0.0', port=5001)