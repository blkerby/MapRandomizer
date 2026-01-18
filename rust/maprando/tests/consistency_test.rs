use std::{path::Path, process::Command};

use anyhow::Result;
use maprando::patch::{Rom, pc2snes};

fn range_subset(inner: (usize, usize), outer: (usize, usize)) -> bool {
    inner.0 >= outer.0 && inner.1 <= outer.1
}

/// Consistency test to ensure that given the same settings and seed values, the same ROM is produced.
/// This helps catch any unintended non-deterministic behavior in the randomization process.
/// This test is marked as ignored by default because it is time-consuming and requires the vanilla ROM,
/// which cannot be distributed.
///
/// The test assumes the vanilla ROM is present at "roms/vanilla.sfc" in the MapRandomizer directory.
#[test]
#[ignore]
fn consistency_test() -> Result<()> {
    let cli_path = env!("CARGO_BIN_EXE_maprando-cli");
    let common_args = [
        "--input-rom",
        "../roms/vanilla.sfc",
        "--map",
        "../maps/vanilla/vanilla_map.json",
        "--random-seed",
        "12345",
    ];

    std::env::set_current_dir("..")?;
    println!("Current directory: {:?}", std::env::current_dir()?);
    let status1 = Command::new(cli_path)
        .args(common_args)
        .args(["--output-rom", "../tmp/consistency_test1.sfc"])
        .status()?;
    assert!(status1.success());

    let status2 = Command::new(cli_path)
        .args(common_args)
        .args(["--output-rom", "../tmp/consistency_test2.sfc"])
        .status()?;
    assert!(status2.success());

    let rom1 = Rom::load(Path::new("../tmp/consistency_test1.sfc"))?;
    let mut rom2 = Rom::load(Path::new("../tmp/consistency_test2.sfc"))?;

    for i in 0..rom1.data.len() {
        if rom1.data[i] != rom2.data[i] {
            rom2.touched.insert(i);
        }
    }
    let mut ranges = vec![];
    for r in rom2.get_modified_ranges() {
        // Ignore differences in the seed name and the resulting checksum:
        let r = (pc2snes(r.0), pc2snes(r.1));
        if range_subset(r, (0xdffef0, 0xdfff00)) || range_subset(r, (0x80ffdc, 0x80ffe0)) {
            continue;
        }
        ranges.push(format!("{:06x}-{:06x}", r.0, r.1));
    }
    if !ranges.is_empty() {
        panic!(
            "Inconsistent ROM generation: differing bytes on {} ranges: {:?}",
            ranges.len(),
            ranges.join(", ")
        );
    }

    Ok(())
}
