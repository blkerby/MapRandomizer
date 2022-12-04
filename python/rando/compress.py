import dataclasses


@dataclasses.dataclass
class RawBlock:
    data: bytes

@dataclasses.dataclass
class RLEBlock:
    size: int
    value: int

BLOCK_RAW = 0
BLOCK_BYTE_RLE = 1

def encode_block_header(size: int, block_type: int, output: bytearray):
    assert size >= 1
    assert size <= 1024
    size1 = size - 1
    if size1 <= 31:
        output.append(size1 | (block_type << 5))
    else:
        output.append(0xE0 | (block_type << 2) | (size1 >> 8))
        output.append(size1 & 0xFF)

def encode_raw_block(data: bytearray, output: bytearray):
    for i in range((len(data) + 1023) // 1024):
        block = data[(i * 1024):((i + 1) * 1024)]
        encode_block_header(len(block), BLOCK_RAW, output)
        output.extend(block)

def encode_rle_block(value: int, count: int, output: bytearray):
    for i in range((count + 1023) // 1024):
        end = min(i * 1024 + 1024, count)
        size = end - i * 1024
        encode_block_header(size, BLOCK_BYTE_RLE, output)
        output.append(value)

def write_block(block_data: bytearray, prev: int, rle_count: int, output: bytearray):
    if len(block_data) >= 1:
        encode_raw_block(block_data, output)
        block_data.clear()
    encode_rle_block(value=prev, count=rle_count, output=output)


# Compress data into format used by Super Metroid for graphics, tilemaps, etc.
# This is a simple encoder that only uses two block types: raw blocks and byte-level RLE (run length encoding).
# Supporting other block types could allow for a smaller compressed output.
def compress(input: bytes):
    output = bytearray()
    block_data = bytearray()
    prev = None  # previous byte value
    rle_count = 0
    for x in input:
        if x == prev:
            rle_count += 1
        else:
            if rle_count >= 3:
                write_block(block_data, prev, rle_count, output)
            else:
                for i in range(rle_count):
                    block_data.append(prev)
            rle_count = 1
            prev = x
    write_block(block_data, prev, rle_count, output)
    output.append(0xFF)
    return output


# import numpy as np
# from debug.decompress import decompress
# import io
# # raw_data = bytes(np.random.randint(0, 1, 15000, dtype=np.uint8))
# # raw_data = bytes(np.random.randint(0, 256, 15000, dtype=np.uint8))
# raw_data = bytes(np.random.randint(0, 2, 15000, dtype=np.uint8))
# compressed_data = compress(raw_data)
# decompressed_data = bytes(decompress(io.BytesIO(compressed_data)))
# print(raw_data == decompressed_data)
# # print(raw_data[:10])
# # print(decompressed_data[:10])