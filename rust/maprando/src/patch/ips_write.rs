#[derive(Clone, Copy)]
struct Chunk {
    start: usize,
    end: usize,
}

fn get_next_chunk(mut pos: usize, old: &[u8], new: &[u8]) -> Chunk {
    loop {
        if pos == old.len() {
            return Chunk {
                start: old.len(),
                end: new.len(),
            };
        }
        if old[pos] != new[pos] {
            break;
        }
        pos += 1;
    }

    let start = pos;
    loop {
        if pos == old.len() {
            return Chunk {
                start,
                end: new.len(),
            };
        }
        if old[pos] == new[pos] {
            break;
        }
        pos += 1;
    }
    Chunk { start, end: pos }
}

fn push_split_chunks(chunk_vec: &mut Vec<Chunk>, chunk: &Chunk) {
    let mut start = chunk.start;
    while start + 0xFFFF < chunk.end {
        chunk_vec.push(Chunk {
            start,
            end: start + 0xFFFF,
        });
        start += 0xFFFF;
    }
    chunk_vec.push(Chunk {
        start,
        end: chunk.end,
    });
}

fn get_chunks(old_rom: &[u8], new_rom: &[u8]) -> Vec<Chunk> {
    let mut pos = 0;
    let mut chunk_vec: Vec<Chunk> = Vec::new();
    loop {
        let chunk = get_next_chunk(pos, old_rom, new_rom);
        if chunk.start != chunk.end {
            push_split_chunks(&mut chunk_vec, &chunk)
        }
        if chunk.end == new_rom.len() {
            return chunk_vec;
        }
        pos = chunk.end;
    }
}

pub fn create_ips_patch(old_rom: &[u8], new_rom: &[u8]) -> Vec<u8> {
    assert!(new_rom.len() >= old_rom.len());
    let mut out: Vec<u8> = Vec::new();
    out.extend("PATCH".as_bytes());
    let chunks = get_chunks(old_rom, new_rom);
    for chunk in &chunks {
        out.extend(&chunk.start.to_be_bytes()[5..8]);
        let size = chunk.end - chunk.start;
        assert!(size <= 0xFFFF); // TODO: Split into sub-chunks if necessary.
        assert!(size > 0);
        out.extend(&size.to_be_bytes()[6..8]);
        out.extend(&new_rom[chunk.start..chunk.end]);
    }
    out.extend("EOF".as_bytes());
    out
}
