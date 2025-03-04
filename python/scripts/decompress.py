# Based on https://patrickjohnston.org/ASM/ROM%20data/Super%20Metroid/decompress.py
import array


def romRead(rom, n=1):
    return int.from_bytes(rom.read(n), 'little')


def decompress(rom, addr, block_type_cnts, block_type_sizes):
    if addr is not None:
        rom.seek(addr)
    decompressed = array.array('B')
    while True:
        byte = romRead(rom)
        if byte == 0xFF:
            break

        type = byte >> 5
        if type != 7:
            size = (byte & 0x1F) + 1
        else:
            size = ((byte & 3) << 8 | romRead(rom)) + 1
            type = byte >> 2 & 7
        # print("{}: len={} type={} size={}".format(rom.tell(), len(decompressed), type, size))

        block_type_cnts[type] += 1
        block_type_sizes[type] += size
        if type == 0:
            decompressed.fromfile(rom, size)
        elif type == 1:
            decompressed.extend([romRead(rom)] * size)
        elif type == 2:
            byte = romRead(rom)
            decompressed.extend([byte, romRead(rom)] * (size >> 1))
            if size & 1:
                decompressed.append(byte)
        elif type == 3:
            byte = romRead(rom)
            decompressed.extend([x % 256 for x in range(byte, byte + size)])
        elif type == 4:
            offset = romRead(rom, 2)
            for i in range(offset, offset + size):
                decompressed.append(decompressed[i])
        elif type == 5:
            offset = romRead(rom, 2)
            for i in range(offset, offset + size):
                decompressed.append(decompressed[i] ^ 0xFF)
        elif type == 6:
            offset = len(decompressed) - romRead(rom)
            for i in range(offset, offset + size):
                decompressed.append(decompressed[i])
        elif type == 7:
            offset = len(decompressed) - romRead(rom)
            for i in range(offset, offset + size):
                decompressed.append(decompressed[i] ^ 0xFF)
    return decompressed


import os
path = "compressed_data/"
files = os.listdir(path)
filename = files[0]

block_type_cnts = {i: 0 for i in range(8)}
block_type_sizes = {i: 0 for i in range(8)}
for filename in files:
    data = open(path + filename, 'rb').read()
    file = open(path + filename, 'rb')
    out = decompress(file, 0, block_type_cnts, block_type_sizes)
    print(len(data), len(out))
    print("cnt:", block_type_cnts)
    print("size:", block_type_sizes)

avg_size = {i: block_type_sizes[i] / block_type_cnts[i] for i in range(8)}
total_size = sum(block_type_sizes.values())
frac = {i: block_type_sizes[i] / total_size for i in range(8)}
print(frac)