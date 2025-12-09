// The changes suggested by this lint usually make the code more cluttered and less clear:
#![allow(clippy::needless_range_loop)]
// TODO: consider removing this later. It's not a bad lint but I don't want to deal with it now.
#![allow(clippy::too_many_arguments)]

pub mod customize;
pub mod difficulty;
pub mod helpers;
pub mod map_repository;
pub mod patch;
pub mod preset;
pub mod randomize;
pub mod seed_repository;
pub mod settings;
pub mod spoiler_log;
pub mod spoiler_map;
pub mod traverse;
