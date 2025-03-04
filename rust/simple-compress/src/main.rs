use anyhow::{bail, Result};
use clap::Parser;
use itertools::Itertools;
use maprando::patch::{decompress, Rom};
use std::fs;
use std::path::PathBuf;
use std::cmp::min;

#[derive(Parser, Debug)]
#[command(about)]
struct Args {
    #[arg(short, long, value_name = "FILE")]
    file: PathBuf, // input file
    #[arg(short, long, value_name = "FILE")]
    output: PathBuf, // output file
    #[arg(short, long)]
    compress: bool,
    #[arg(short, long)]
    decompress: bool,
}

const BLOCK_TYPE_DIRECT: u8 = 0;
const BLOCK_TYPE_WORD_RLE: u8 = 2;

fn bytes_to_words(input_bytes: &[u8]) -> Vec<u16> {
    let mut i = 0;
    let mut out: Vec<u16> = vec![];
    while i < input_bytes.len() {
        if i + 1 == input_bytes.len() {
            out.push(input_bytes[i] as u16);
        } else {
            out.push(input_bytes[i] as u16 | ((input_bytes[i + 1] as u16) << 8));
        }
        i += 2;
    }
    out
}

fn words_to_bytes(input_words: &[u16]) -> Vec<u8> {
    let mut out: Vec<u8> = vec![];
    for &w in input_words {
        out.push((w & 0xFF) as u8);
        out.push((w >> 8) as u8);
    }
    out
}

fn encode_block_header(size: usize, block_type: u8, out: &mut Vec<u8>) {
    assert!(size >= 1);
    assert!(size <= 1024);
    let size1 = size - 1;
    if size1 <= 31 {
        out.push(size1 as u8 | (block_type << 5));
    } else {
        out.push(0xE0 | (block_type << 2) | ((size1 >> 8) as u8));
        out.push((size1 & 0xFF) as u8);
    }
}

fn encode_direct_block(data: &[u8], out: &mut Vec<u8>) {
    for i in 0..((data.len() + 1023) / 1024) {
        let block = &data[(i * 1024)..min((i + 1) * 1024, data.len())];
        encode_block_header(block.len(), BLOCK_TYPE_DIRECT, out);
        out.extend(block);
    }
}

fn encode_word_rle_block(value: u16, count: usize, out: &mut Vec<u8>) {
    for i in 0..((count + 511) / 512) {
        let end = min(i * 512 + 512, count);
        let size = end - i * 512;
        encode_block_header(size * 2, BLOCK_TYPE_WORD_RLE, out);
        out.push((value & 0xFF) as u8);
        out.push((value >> 8) as u8);
    }
}

fn compress(input_bytes: &[u8]) -> Vec<u8> {
    let input_words: Vec<u16> = bytes_to_words(input_bytes);
    let mut out: Vec<u8> = vec![];
    let mut buf: Vec<u16> = vec![];
    for (&value, chunk) in &input_words.iter().chunk_by(|&x| x) {
        let chunk_vec: Vec<u16> = chunk.copied().collect();
        if chunk_vec.len() >= 3 {
            if buf.len() >= 1 {
                let buf_bytes = words_to_bytes(&buf);
                encode_direct_block(&buf_bytes, &mut out);
                buf.clear();
            }
            encode_word_rle_block(value, chunk_vec.len(), &mut out);
        } else {
            buf.extend(chunk_vec);
        }
    }
    if buf.len() >= 1 {
        let buf_bytes = words_to_bytes(&buf);
        encode_direct_block(&buf_bytes, &mut out);
    }
    out.push(0xFF);
    out
}

fn main() -> Result<()> {
    let args = Args::parse();
    // println!("{:?}", args);
    let compressed_input_bytes = fs::read(args.file)?;
    let input_rom = Rom::new(compressed_input_bytes.clone());
    let input_bytes = if args.decompress {
        decompress::decompress(&input_rom, 0)?
    } else {
        compressed_input_bytes.clone()
    };
    let compressed_output_bytes = compress(&input_bytes);
    let output_rom = Rom::new(compressed_output_bytes.clone());
    let decompressed_output_bytes = decompress::decompress(&output_rom, 0)?;
    if input_bytes != decompressed_output_bytes {
        bail!("Decompressed output fails to match input");
    }
    // println!(
    //     "original input={}, decompressed={}, output={}",
    //     compressed_input_bytes.len(),
    //     input_bytes.len(),
    //     compressed_output_bytes.len()
    // );
    fs::write(args.output, compressed_output_bytes)?;
    Ok(())
}
