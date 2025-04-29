import os
import glob
import json
import logging
import argparse

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Build the IPS patches without needing a vanilla ROM:
# We run assembler twice for each patch, once with a ROM filled with $00, and once with a ROM filled with $ff,
# to be able to detect which bytes are affected.

parser = argparse.ArgumentParser('build_ips', 'Build IPS files')
parser.add_argument('--rebuild', action=argparse.BooleanOptionalAction)
parser.add_argument('--verify', action=argparse.BooleanOptionalAction)
parser.add_argument('--assembler-path', type=str, default="asar")
args = parser.parse_args()

TMP_PATH = "tmp"
ROM_PATH = TMP_PATH + "/rom.sfc"
OUTPUT_PATH = "patches/ips"

excluded_files = [
    "Mosaic/Projects/Base/ASM/Acid Tilemap.asm",
    "Mosaic/Projects/Base/ASM/Area Map Debug.asm",
]
ignored_overlap_patterns = ["hyper_beam", "ultra_low_"]

asm_src_files = []
asm_src_files.extend(glob.glob("patches/src/*.asm"))
asm_src_files.extend(glob.glob("Mosaic/Projects/Base/ASM/*.asm"))
asm_src_files = [x for x in asm_src_files if x not in excluded_files]

os.makedirs(TMP_PATH, exist_ok=True)

snes2pc = lambda address: address >> 1 & 0x3F8000 | address & 0x7FFF
pc2snes = lambda address: address << 1 & 0xFF0000 | address & 0xFFFF | 0x808000

def run_asar(asm_path, fill_byte, changed_byte_dict):
    # Initialize the ROM using the fill byte
    rom_data = bytes(0x400000 * [fill_byte])
    rom_file = open(ROM_PATH, 'wb')
    rom_file.write(rom_data)
    rom_file.close()
    
    # Run the assembler
    cmd = f'{args.assembler_path} --no-title-check --fix-checksum=off "{asm_path}" "{ROM_PATH}"'
    # print("> " + cmd)
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError("Assembler command failed: " + cmd)
    
    rom_data = open(ROM_PATH, 'rb').read()
    for i, x in enumerate(rom_data):
        if x != fill_byte:
            changed_byte_dict[i] = x


def get_chunks(changed_byte_dict):
    start = None
    last = None
    chunks = []
    for i in sorted(changed_byte_dict.keys()):
        if start is None:
            start = i
            last = i
        elif i == last + 1:
            last = i
        else:
            chunks.append((start, last + 1))
            start = i
            last = i
    if start is not None:
        chunks.append((start, last + 1))
    return chunks


def write_ips_patch(ips_path, changed_byte_dict, chunks):
    file = open(ips_path, 'wb')
    file.write("PATCH".encode())
    for (start, end) in chunks:
        size = end - start
        assert size > 0
        assert size <= 0xFFFF  # TODO: Split into sub-chunks if necessary.
        file.write(start.to_bytes(3, 'big'))
        file.write(size.to_bytes(2, 'big'))
        data = bytes(changed_byte_dict[i] for i in range(start, end))
        file.write(data)
    file.write("EOF".encode());


def read_ips_patch_chunks(ips_path):
    file = open(ips_path, 'rb')
    assert file.read(5) == "PATCH".encode()
    chunks = []
    while True:
        start_bytes = file.read(3)
        if start_bytes == "EOF".encode():
            break
        start = int.from_bytes(start_bytes, byteorder="big")
        size = int.from_bytes(file.read(2), byteorder="big")
        file.read(size)
        chunks.append((start, start + size))
    return chunks

def check_range(filename, other_filename, overlap_start, overlap_end):
    val_dict = whitelist_dict.get(filename)
    for key in val_dict:
        if pc2snes(overlap_start) >= key[1] and pc2snes(overlap_end) <= key[2]:
            return True
    return False

# whitelist of intended overlapping patches
whitelist_dict = {
    'etank_refill_full': [
        ['etank_refill_disabled', 0x848972, 0x848975]
    ],
    'map_area': [
        ['hud_expansion_opaque', 0x809DBF, 0x809DE7],
        ['item_dots_disappear', 0x90AA73, 0x90AA7B],
    ],
    'fast_pause_menu': [
        ['pause_menu_objectives', 0x82932E, 0x829332]
    ],
    'map_progress_maintain': [
        ['map_area', 0x829440, 0x829443],
        ['map_area', 0x829475, 0x829478],
        ['map_area', 0x829ED5, 0x829ED8],
        ['item_dots_disappear', 0x82945C, 0x82945F]
    ],
    'pause_menu_objectives': [
        ['map_area', 0x82910A, 0x82910D],
        ['hud_expansion_opaque', 0xB69A00, 0xB6AD40]
    ],
    'hud_expansion_opaque': [
        ['max_ammo_display_fast', 0x858851, 0x858A93],
        ['max_ammo_display_fast', 0x9AB542, 0x9AB6C0],
        ['Area Palettes', 0x82E569, 0x82E56C]
    ]
}

failed = False
chunks_dict = {}
for idx, asm_path in enumerate(asm_src_files):
    base_filename = os.path.splitext(os.path.basename(asm_path))[0]
    ips_path = f"{OUTPUT_PATH}/{base_filename}.ips"
    
    src_modified_ts = os.path.getmtime(asm_path)
    ips_modified_ts = None
    try:
        ips_modified_ts = os.path.getmtime(ips_path)
    except FileNotFoundError as e:
        pass
    
    if ips_modified_ts is None or src_modified_ts > ips_modified_ts or args.rebuild or args.verify:
        logging.info(f"Assembling {asm_path}")
        changed_bytes = {}
        run_asar(asm_path, 0x00, changed_bytes)
        run_asar(asm_path, 0xff, changed_bytes)
        chunks = get_chunks(changed_bytes)

        # Write the IPS file:
        if args.verify:
            patch_before = open(ips_path, "rb").read()
            write_ips_patch(ips_path, changed_bytes, chunks)
            patch_after = open(ips_path, "rb").read()
            if patch_before != patch_after:
                logging.error("🔴 IPS patch is out-of-date: " + base_filename)
                failed = True
        else:
            write_ips_patch(ips_path, changed_bytes, chunks)
    else:
        chunks = read_ips_patch_chunks(ips_path)
    chunks_dict[base_filename] = chunks


# Check for overlap/conflicts between patches:
for filename, chunks in chunks_dict.items():
    for other_filename, other_chunks in chunks_dict.items():
        if filename == other_filename:
            continue
        ignored = False
        for p in ignored_overlap_patterns:
            if p in filename or p in other_filename:
                ignored = True
        if ignored:
            continue
        for chunk in chunks:
            for other_chunk in other_chunks:
                overlap_start = max(chunk[0], other_chunk[0])
                overlap_end = min(chunk[1], other_chunk[1])
                if overlap_start < overlap_end:
                    ignored = False
                    if filename in whitelist_dict:
                        ignored = check_range(filename, other_filename, overlap_start, overlap_end)
                    if ignored is False and other_filename in whitelist_dict:
                        ignored = check_range(other_filename, filename, overlap_start, overlap_end)
                    if not ignored:
                        logging.error(f"🔴 Overlap between {filename} and {other_filename}: {pc2snes(overlap_start):06X} - {pc2snes(overlap_end):06X}")
                        failed = True

if failed:
    os._exit(1)
