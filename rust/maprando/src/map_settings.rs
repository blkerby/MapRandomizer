use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::settings::{AreaAssignment, AreaAssignmentPreset, MapPreset};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AreaPreferencePreset {
    Unrestricted,
    Concentrated,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AreaPreference {
    pub preset: Option<AreaPreferencePreset>,
    pub probabilities: [f64; 3],
}

impl AreaPreference {
    pub fn from_preset(preset: AreaPreferencePreset, area_share: f64) -> Self {
        Self {
            preset: Some(preset),
            probabilities: match preset {
                AreaPreferencePreset::Unrestricted => [area_share; 3],
                AreaPreferencePreset::Concentrated => [0.5, 0.6, 0.7],
            },
        }
    }

    pub fn effective_probabilities(&self, area_share: f64) -> [f64; 3] {
        if self.preset == Some(AreaPreferencePreset::Unrestricted) {
            [area_share; 3]
        } else {
            self.probabilities
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Landmark {
    Ship,
    Kraid,
    Ridley,
    Phantoon,
    Draygon,
    MotherBrain,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LandmarksPreset {
    None,
    ShipMB,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AreaLocationsPreset {
    Unrestricted,
    Preferred,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ReturnPathsPreset {
    Unrestricted,
    Short,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AreaSizesPreset {
    Unbiased,
    Familiar,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapGenerationSettings {
    pub small_map: bool,
    pub landmarks_preset: Option<LandmarksPreset>,
    pub landmarks: Vec<Landmark>,
    pub maridia_water: AreaPreference,
    pub norfair_heat: AreaPreference,
    pub reward_missing_connect_utility: f64,
    pub door_balance_strength: f64,
    pub return_paths_preset: Option<ReturnPathsPreset>,
    pub max_return_distance: Option<f64>,
    pub area_locations_preset: Option<AreaLocationsPreset>,
    pub reward_area_x: f64,
    pub reward_area_y: f64,
    pub area_sizes_preset: Option<AreaSizesPreset>,
    pub target_area_rooms: [f64; 6],
    pub target_area_x: [f64; 6],
    pub target_area_y: [f64; 6],
    pub reward_graph_diameter: f64,
    pub reward_save_distance: f64,
    pub reward_refill_distance: f64,
    pub reward_area_distinct_crossing: f64,
}

impl MapGenerationSettings {
    pub fn apply_presets(&mut self) {
        if let Some(preset) = self.landmarks_preset {
            self.landmarks = match preset {
                LandmarksPreset::None => vec![],
                LandmarksPreset::ShipMB => vec![Landmark::Ship, Landmark::MotherBrain],
                LandmarksPreset::All => vec![
                    Landmark::Ship,
                    Landmark::Kraid,
                    Landmark::Ridley,
                    Landmark::Phantoon,
                    Landmark::Draygon,
                    Landmark::MotherBrain,
                ],
            };
        }
        if let Some(preset) = self.return_paths_preset {
            (
                self.reward_missing_connect_utility,
                self.max_return_distance,
            ) = match preset {
                ReturnPathsPreset::Unrestricted => (0.0, None),
                ReturnPathsPreset::Short => (2.0, Some(11.0)),
            };
        }
        if let Some(preset) = self.area_locations_preset {
            let preferred = preset == AreaLocationsPreset::Preferred;
            self.reward_area_x = if preferred { 10.0 } else { 0.0 };
            self.reward_area_y = if preferred { 10.0 } else { 0.0 };
            self.target_area_x = if preferred {
                [16.0, 10.0, 40.0, 48.0, 40.0, 16.0]
            } else {
                [36.0; 6]
            };
            self.target_area_y = if preferred {
                [12.0, 32.0, 64.0, 12.0, 32.0, 24.0]
            } else {
                [36.0; 6]
            };
        }
        if let Some(preset) = self.area_sizes_preset {
            self.target_area_rooms = match preset {
                AreaSizesPreset::Unbiased => [1.0; 6],
                AreaSizesPreset::Familiar => [150.0, 200.0, 200.0, 100.0, 200.0, 100.0],
            };
        }
        if let Some(preset) = self.maridia_water.preset {
            self.maridia_water = AreaPreference::from_preset(preset, self.area_share(4));
        }
        if let Some(preset) = self.norfair_heat.preset {
            self.norfair_heat = AreaPreference::from_preset(preset, self.area_share(2));
        }
    }

    pub fn area_share(&self, area: usize) -> f64 {
        self.target_area_rooms[area] / self.target_area_rooms.iter().sum::<f64>()
    }

    pub fn validate(&self) -> Result<()> {
        for weight in self.target_area_rooms {
            if !weight.is_finite() || weight <= 0.0 {
                bail!("Area size weights must be finite and greater than zero");
            }
        }
        if !self.target_area_rooms.iter().sum::<f64>().is_finite() {
            bail!("Total area size weight must be finite");
        }
        for coordinate in self.target_area_x.iter().chain(&self.target_area_y) {
            if !coordinate.is_finite() {
                bail!("Area center coordinates must be finite");
            }
        }
        for (name, value) in [
            (
                "Return-path preference",
                self.reward_missing_connect_utility,
            ),
            ("Door balance strength", self.door_balance_strength),
            ("Horizontal position reward", self.reward_area_x),
            ("Vertical position reward", self.reward_area_y),
            ("Graph diameter reward", self.reward_graph_diameter),
            ("Save distance reward", self.reward_save_distance),
            ("Refill distance reward", self.reward_refill_distance),
        ] {
            if !value.is_finite() {
                bail!("{name} must be finite");
            }
        }
        if !self.reward_area_distinct_crossing.is_finite() {
            bail!("Distinct area crossings reward must be finite");
        }
        if let Some(distance) = self.max_return_distance
            && (!distance.is_finite() || distance < 0.0)
        {
            bail!("Max return distance must be finite and nonnegative");
        }
        for (name, preference, area) in [
            ("Water in Maridia", self.maridia_water, 4),
            ("Heat in Norfair", self.norfair_heat, 2),
        ] {
            let baseline = self.area_share(area);
            let p = preference.effective_probabilities(baseline);
            if p.iter().any(|x| !x.is_finite())
                || !(0.0 < p[0] && baseline <= p[0] && p[0] <= p[1] && p[1] <= p[2] && p[2] < 1.0)
            {
                bail!(
                    "{name} probabilities must be positive, nondecreasing, below one, and at least the area's size share ({baseline:.4})"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapSettings {
    pub preset: Option<MapPreset>,
    pub source: String,
    pub generation: MapGenerationSettings,
    pub area_assignment: Option<AreaAssignment>,
}

impl MapSettings {
    pub fn from_preset(preset: MapPreset) -> Self {
        let concentrated = preset == MapPreset::Familiar;
        let climate = AreaPreference::from_preset(
            if concentrated {
                AreaPreferencePreset::Concentrated
            } else {
                AreaPreferencePreset::Unrestricted
            },
            if preset == MapPreset::Wild {
                1.0 / 6.0
            } else {
                200.0 / 950.0
            },
        );
        let source = match preset {
            MapPreset::Vanilla => "vanilla",
            _ => "Fresh",
        };
        let mut settings = Self {
            preset: Some(preset),
            source: source.to_string(),
            generation: MapGenerationSettings {
                small_map: preset == MapPreset::Small,
                landmarks_preset: Some(match preset {
                    MapPreset::Familiar => LandmarksPreset::All,
                    MapPreset::Standard => LandmarksPreset::ShipMB,
                    _ => LandmarksPreset::None,
                }),
                landmarks: vec![],
                maridia_water: climate,
                norfair_heat: climate,
                reward_missing_connect_utility: 2.0,
                door_balance_strength: if matches!(
                    preset,
                    MapPreset::Small | MapPreset::Familiar | MapPreset::Standard
                ) {
                    0.3
                } else {
                    1.0
                },
                return_paths_preset: if matches!(
                    preset,
                    MapPreset::Small | MapPreset::Familiar | MapPreset::Standard
                ) {
                    Some(ReturnPathsPreset::Short)
                } else if preset == MapPreset::Wild {
                    Some(ReturnPathsPreset::Unrestricted)
                } else {
                    None
                },
                max_return_distance: None,
                area_locations_preset: Some(if concentrated {
                    AreaLocationsPreset::Preferred
                } else {
                    AreaLocationsPreset::Unrestricted
                }),
                reward_area_x: 0.0,
                reward_area_y: 0.0,
                area_sizes_preset: Some(if preset == MapPreset::Wild {
                    AreaSizesPreset::Unbiased
                } else {
                    AreaSizesPreset::Familiar
                }),
                target_area_rooms: [1.0; 6],
                target_area_x: [36.0; 6],
                target_area_y: [36.0; 6],
                reward_graph_diameter: 0.2,
                reward_save_distance: 0.2,
                reward_refill_distance: 0.1,
                reward_area_distinct_crossing: 1.0,
            },
            area_assignment: None,
        };
        settings.apply_subpresets();
        settings
    }

    pub fn apply_subpresets(&mut self) {
        self.generation.apply_presets();
        if let Some(assignment) = &mut self.area_assignment
            && let Some(preset) = assignment.preset
        {
            *assignment = AreaAssignment::from_preset(preset);
        }
    }

    pub fn is_vanilla(&self) -> bool {
        self.source == "vanilla"
    }

    pub fn is_small(&self) -> bool {
        !self.is_vanilla() && self.generation.small_map
    }

    pub fn label(&self) -> String {
        self.preset
            .map(|p| format!("{p:?}"))
            .unwrap_or_else(|| "Custom".to_string())
    }

    pub fn validate(&self, fresh_available: bool) -> Result<()> {
        if self.source == "Fresh" {
            if !fresh_available {
                bail!("Fresh map generation is not configured on this server");
            }
            self.generation.validate()?;
        }
        Ok(())
    }
}

pub fn map_subpresets() -> serde_json::Value {
    use serde_json::json;

    let base = MapSettings::from_preset(MapPreset::Familiar).generation;
    json!({
        "landmarks": ([LandmarksPreset::None, LandmarksPreset::ShipMB, LandmarksPreset::All].map(|preset| {
            let mut g = base.clone();
            g.landmarks_preset = Some(preset);
            g.apply_presets();
            json!({"preset": preset, "landmarks": g.landmarks})
        })),
        "locations": ([AreaLocationsPreset::Unrestricted, AreaLocationsPreset::Preferred].map(|preset| {
            let mut g = base.clone();
            g.area_locations_preset = Some(preset);
            g.apply_presets();
            json!({"preset": preset, "reward_area_x": g.reward_area_x, "reward_area_y": g.reward_area_y,
                "target_area_x": g.target_area_x, "target_area_y": g.target_area_y})
        })),
        "loops": ([ReturnPathsPreset::Unrestricted, ReturnPathsPreset::Short].map(|preset| {
            let mut g = base.clone();
            g.return_paths_preset = Some(preset);
            g.apply_presets();
            json!({"preset": preset, "reward_missing_connect_utility": g.reward_missing_connect_utility,
                "max_return_distance": g.max_return_distance})
        })),
        "sizes": ([AreaSizesPreset::Unbiased, AreaSizesPreset::Familiar].map(|preset| {
            let mut g = base.clone();
            g.area_sizes_preset = Some(preset);
            g.apply_presets();
            json!({"preset": preset, "target_area_rooms": g.target_area_rooms})
        })),
        "climate": ([AreaPreferencePreset::Unrestricted, AreaPreferencePreset::Concentrated]
            .map(|preset| AreaPreference::from_preset(preset, base.area_share(4)))),
        "relabel": ([AreaAssignmentPreset::Standard, AreaAssignmentPreset::Size,
            AreaAssignmentPreset::Depth, AreaAssignmentPreset::Random].map(AreaAssignment::from_preset)),
    })
}
