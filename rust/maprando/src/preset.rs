use maprando_game::TechId;
use serde::{Deserialize, Serialize};

// #[derive(Serialize, Deserialize, Clone)]
// pub struct TechSetting {
//     tech_id: usize,
//     name: String,
// }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TechData {
    pub tech_id: TechId,
    pub name: String,
    pub video_id: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NotableData {
    pub room_id: usize,
    pub notable_id: usize,
    pub room_name: String,
    pub name: String,
    pub video_id: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Preset {
    pub name: String,
    pub shinespark_tiles: f32,
    pub heated_shinespark_tiles: f32,
    pub speed_ball_tiles: f32,
    pub shinecharge_leniency_frames: usize,
    pub resource_multiplier: f32,
    pub escape_timer_multiplier: f32,
    pub gate_glitch_leniency: usize,
    pub door_stuck_leniency: usize,
    pub phantoon_proficiency: f32,
    pub draygon_proficiency: f32,
    pub ridley_proficiency: f32,
    pub botwoon_proficiency: f32,
    pub mother_brain_proficiency: f32,
    pub tech: Vec<TechData>,
    pub notables: Vec<NotableData>,
}
