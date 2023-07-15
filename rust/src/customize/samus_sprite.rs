use std::{io::Cursor, collections::HashMap, path::Path, cmp::min, cmp::max};
use std::iter;

use anyhow::{Result, Context};
use image::io::Reader as ImageReader;
use serde_derive::Deserialize;

use crate::patch::Rom;


const SPRITESHEET_WIDTH: isize = 876;
const SPRITESHEET_HEIGHT: isize = 2543;

#[derive(Deserialize, Debug, Clone)]
struct SpritesheetImage {
    parent: Option<String>,
    dimensions: Option<[isize; 4]>,
    #[serde(rename="extra area")]
    extra_area: Option<Vec<[isize; 4]>>,
    scale: Option<isize>,
    shift: Option<(isize, isize)>,
    spacer: Option<isize>,
    palette: Option<String>,
    #[serde(rename="import palette interval")]
    import_palette_interval: Option<(usize, usize)>,
    #[serde(rename="used by")]
    used_by: Vec<(String, usize)>,
}

#[derive(Deserialize, Debug)]
struct SpritesheetLayout {
    layout: Vec<Vec<String>>,
    images: HashMap<String, SpritesheetImage>,
    dma_sequence: Vec<String>,
}

pub struct SamusSpriteCustomizer {
    spritesheet_layout: SpritesheetLayout,
}

fn populate_from_parent(image_name: &str, images: &mut HashMap<String, SpritesheetImage>) {
    let parent_name = images.get_mut(image_name).unwrap().parent.clone();
    if let Some(parent_name) = parent_name.as_ref() {
        populate_from_parent(parent_name, images);
        let parent = images.get(parent_name).unwrap().clone();
        let mut img = images.get_mut(image_name).unwrap().clone();
        *images.get_mut(image_name).unwrap() = SpritesheetImage {
            parent: img.parent,
            dimensions: img.dimensions.or(parent.dimensions),
            extra_area: img.extra_area.or(parent.extra_area),
            scale: img.scale.or(parent.scale),
            shift: img.shift.or(parent.shift),
            spacer: img.spacer.or(parent.spacer),
            palette: img.palette.or(parent.palette),
            import_palette_interval: img.import_palette_interval.or(parent.import_palette_interval),
            used_by: img.used_by,
        };
    }
}

fn get_local_image_bounding_box(image: &SpritesheetImage) -> [isize; 4] {
    let [mut x_min, mut y_min, mut x_max, mut y_max] = image.dimensions.unwrap();
    if let Some(extra_area) = image.extra_area.as_ref() {
        for [extra_x_min, extra_y_min, extra_x_max, extra_y_max] in extra_area {
            x_min = min(x_min, *extra_x_min);
            y_min = min(y_min, *extra_y_min);
            x_max = max(x_max, *extra_x_max);
            y_max = max(y_max, *extra_y_max);
        }
    }
    if let Some(scale) = image.scale {
        x_min *= scale;
        x_max *= scale;
        y_min *= scale;
        y_max *= scale;
    }
    if let Some((shift_x, shift_y)) = image.shift {
        x_min += shift_x;
        x_max += shift_x;
        y_min += shift_y;
        y_max += shift_y;
    }
    [x_min, y_min, x_max, y_max]
}

fn get_row_dims(image_names: &[String], images: &HashMap<String, SpritesheetImage>) -> (isize, isize, isize) {
    let mut y_min = isize::MAX;
    let mut y_max = isize::MIN;
    let mut width = 0;

    for name in image_names {
        let img = &images[name];
        let b = get_local_image_bounding_box(img);
        y_min = min(y_min, b[1]);
        y_max = max(y_max, b[3]);
        width += b[2] - b[0] + 2 + img.spacer.unwrap_or(0).abs();
    }
    (width, y_min, y_max)
}

fn get_image_bounding_boxes(layout: &SpritesheetLayout) -> HashMap<String, [isize; 4]> {
    let mut out: HashMap<String, [isize; 4]> = HashMap::new();

    let mut y = 0;
    for row in &layout.layout {
        let (row_width, row_y_min, row_y_max) = get_row_dims(row, &layout.images);
        let mut x = (SPRITESHEET_WIDTH - row_width) / 2;
        for image_name in row {
            let img = &layout.images[image_name];
            let b = get_local_image_bounding_box(img);
            let shift_y = img.shift.unwrap_or((0, 0)).1;
            let spacer = img.spacer.unwrap_or(0);
            x -= min(spacer, 0);
            let img_x_min = x + 1;
            let img_x_max = x + 1 + b[2] - b[0];
            let img_y_min = y + 1 + b[1] - row_y_min + shift_y;
            let img_y_max = y + 1 + b[3] - row_y_min + shift_y;
            out.insert(image_name.clone(), [img_x_min, img_y_min, img_x_max, img_y_max]);
            // println!("{}: {} {} {} {}", image_name, img_x_min, img_y_min, img_x_max, img_y_max);
            x += b[2] - b[0] + max(spacer, 0) + 2;
        }
        y += row_y_max - row_y_min + 2;
    }
    out
}

#[derive(Debug)]
struct TilePosition {
    large: bool,  // True if 16 x 16 tile (else 8 x 8)
    // Coordinates of where to draw the tile
    draw_x: isize,
    draw_y: isize,
    // Coordinates of upper-left corner of tile within spritesheet
    sheet_x: isize,
    sheet_y: isize,
}

fn get_image_tile_positions(bounding_box: [isize; 4], image: &SpritesheetImage) -> Vec<TilePosition> {
    let mut large_tiles = vec![];
    let mut small_tiles = vec![];
    for b in iter::once(image.dimensions.unwrap()).chain(image.extra_area.clone().unwrap_or(vec![])) {
        for y in (b[1]..b[3]).step_by(16) {
            for x in (b[0]..b[2]).step_by(16) {
                if b[3] - y >= 16 {
                    if b[2] - x >= 16 {
                        large_tiles.push((x, y));
                    } else {
                        small_tiles.push((x, y));
                        small_tiles.push((x, y + 8));
                    }
                } else {
                    if b[2] - x >= 16 {
                        small_tiles.push((x, y));                    
                        small_tiles.push((x + 8, y));
                    } else {
                        small_tiles.push((x, y));
                    }
                }
            }
        }
    }
    let mut out: Vec<TilePosition> = vec![];
    for t in &large_tiles {
        let b = get_local_image_bounding_box(image);
        out.push(TilePosition {
            large: true,
            draw_x: t.0,
            draw_y: t.1,
            sheet_x: bounding_box[0] + t.0 - b[0],
            sheet_y: bounding_box[1] + t.1 - b[1],
        })
    }
    out
}

impl SamusSpriteCustomizer {
    pub fn new(spritesheet_layout_path: &Path) -> Result<Self> {
        let spritesheet_layout_str = std::fs::read_to_string(spritesheet_layout_path)?;
        let mut spritesheet_layout: SpritesheetLayout = serde_json::from_str(&spritesheet_layout_str)?;
        let image_names: Vec<String> = spritesheet_layout.images.keys().cloned().collect();
        for image_name in &image_names {
            populate_from_parent(image_name, &mut spritesheet_layout.images);
        }

        let image_boxes: HashMap<String, [isize; 4]> = get_image_bounding_boxes(&spritesheet_layout);
        for image_name in &spritesheet_layout.dma_sequence {
            let image = &spritesheet_layout.images[image_name];
            let tiles = get_image_tile_positions(image_boxes[image_name], image);
            println!("{}: {:?}", image_name, tiles);
        }

        Ok(SamusSpriteCustomizer { spritesheet_layout })
    }

    pub fn apply(&self, rom: &mut Rom, sprite_bytes: &[u8]) -> Result<()> {
        let img_raw = ImageReader::with_format(Cursor::new(sprite_bytes), image::ImageFormat::Png).decode()?;
        let img = img_raw.as_rgba8().context("expecting RGBA8 image")?;
        // img.
        Ok(())
    }
    
}

