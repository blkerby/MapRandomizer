# Based on https://patrickjohnston.org/ASM/ROM%20data/Super%20Metroid/decompress.py
import array


def romRead(rom, n=1):
    return int.from_bytes(rom.read(n), 'little')


def decompress(rom, addr=None):
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
            # Not handling 8-bit overflow (byte + size > 256)
            byte = romRead(rom)
            decompressed.extend(range(byte, byte + size))
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