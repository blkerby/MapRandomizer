use anyhow::{Context, Result};
use clap::Parser;
use crypto_hash;
use maprando::game_data::smart_xml;
use maprando::game_data::themed_retiling::extract_uncompressed_level_data;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    compressor: PathBuf,
}

fn write_compressed_data(
    output_dir: &Path,
    compressor_path: &Path,
    tmp_dir: &Path,
    data: &[u8],
    data_name: &str,
) -> Result<()> {
    let digest = crypto_hash::hex_digest(crypto_hash::Algorithm::SHA256, &data);
    let output_path = output_dir.join(digest);
    let tmp_path = tmp_dir.join("tmpfile");
    if output_path.exists() {
        println!("Skipping {}", data_name);
        return Ok(());
    } else {
        println!("Processing {}", data_name);
    }
    std::fs::write(&tmp_path, data)?;
    Command::new(compressor_path)
        .arg("-c")
        .arg(format!("-f={}", tmp_path.to_str().unwrap()))
        .arg(format!("-o={}", output_path.to_str().unwrap()))
        .status()
        .context("error running compressor")?;
    Ok(())
}

fn write_compressed_file(
    output_dir: &Path,
    compressor_path: &Path,
    tmp_path: &Path,
    data_path: &Path,
) -> Result<()> {
    let data = std::fs::read(data_path)
        .with_context(|| format!("error reading {}", data_path.display()))?;
    write_compressed_data(output_dir, compressor_path, tmp_path, &data, data_path.to_str().unwrap())
        .with_context(|| format!("error compressing {}", data_path.display()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let compressor_path = &args.compressor;
    let output_dir = Path::new("../compressed_data");
    let mosaic_dir = Path::new("../Mosaic");
    let tmp_dir = Path::new("../tmp");
    std::fs::create_dir_all(tmp_dir)?;
    std::fs::create_dir_all(output_dir)?;
    let projects_path = mosaic_dir.join("Projects");
    for project_dir in std::fs::read_dir(projects_path)? {
        let project_dir = project_dir?;

        // Process CRE tileset:
        let cre_path = project_dir.path().join("Export/Tileset/CRE/00");
        let gfx8x8_path = cre_path.join("8x8tiles.gfx");
        write_compressed_file(output_dir, compressor_path, tmp_dir, &gfx8x8_path)?;
        let gfx16x16_path = cre_path.join("16x16tiles.ttb");
        write_compressed_file(output_dir, compressor_path, tmp_dir, &gfx16x16_path)?;

        // Process SRE tilesets:
        let sce_path = project_dir.path().join("Export/Tileset/SCE");
        for tileset_path in std::fs::read_dir(sce_path)? {
            let tileset_path = tileset_path?.path();

            let palette_path = tileset_path.join("palette.snes");
            write_compressed_file(output_dir, compressor_path, tmp_dir, &palette_path)?;

            let gfx8x8_path = tileset_path.join("8x8tiles.gfx");
            write_compressed_file(output_dir, compressor_path, tmp_dir, &gfx8x8_path)?;

            let gfx16x16_path = tileset_path.join("16x16tiles.ttb");
            write_compressed_file(output_dir, compressor_path, tmp_dir, &gfx16x16_path)?;
        }

        // Process rooms:
        let room_dir = project_dir.path().join("Export/Rooms");
        for room_path in std::fs::read_dir(room_dir)? {
            let room_path = room_path?.path();
            let room_str = std::fs::read_to_string(&room_path)?;
            let room: smart_xml::Room = serde_xml_rs::from_str(room_str.as_str()).unwrap();
            for state in &room.states.state {
                let level_data = extract_uncompressed_level_data(state);
                write_compressed_data(output_dir, compressor_path, tmp_dir, &level_data, room_path.to_str().unwrap())?;
            }
        }        
    }
    Ok(())
}
