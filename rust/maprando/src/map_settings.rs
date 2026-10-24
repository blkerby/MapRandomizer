use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::settings::{AreaAssignment, AreaAssignmentPreset, MapPreset};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum MapSource {
    Vanilla,
    Small,
    Standard,
    Wild,
    Fresh,
}

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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapGenerationSettings {
    pub small_map: bool,
    pub landmarks: Vec<Landmark>,
    pub maridia_water: AreaPreference,
    pub norfair_heat: AreaPreference,
    pub reward_missing_connect_utility: f64,
    pub max_return_distance: Option<f64>,
    pub reward_area_x: f64,
    pub reward_area_y: f64,
    pub target_area_rooms: [f64; 6],
    pub target_area_x: [f64; 6],
    pub target_area_y: [f64; 6],
    pub reward_graph_diameter: f64,
    pub reward_save_distance: f64,
    pub reward_refill_distance: f64,
    pub reward_area_distinct_crossing: f64,
}

impl MapGenerationSettings {
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
            ("Horizontal position reward", self.reward_area_x),
            ("Vertical position reward", self.reward_area_y),
            ("Graph diameter reward", self.reward_graph_diameter),
            ("Save distance reward", self.reward_save_distance),
            ("Refill distance reward", self.reward_refill_distance),
        ] {
            if !value.is_finite() || value < 0.0 {
                bail!("{name} must be finite and nonnegative");
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
    pub source: MapSource,
    pub generation: MapGenerationSettings,
    pub area_assignment: Option<AreaAssignment>,
}

impl MapSettings {
    pub fn from_preset(preset: MapPreset, fresh_available: bool) -> Self {
        let preferred = matches!(preset, MapPreset::Standard | MapPreset::Familiar);
        let concentrated = preset == MapPreset::Familiar;
        let climate = AreaPreference {
            preset: Some(if concentrated {
                AreaPreferencePreset::Concentrated
            } else {
                AreaPreferencePreset::Unrestricted
            }),
            probabilities: if concentrated {
                [0.5, 0.6, 0.7]
            } else if preset == MapPreset::Wild {
                [1.0 / 6.0; 3]
            } else {
                [200.0 / 950.0; 3]
            },
        };
        let source = match preset {
            MapPreset::Vanilla => MapSource::Vanilla,
            _ if fresh_available => MapSource::Fresh,
            MapPreset::Small => MapSource::Small,
            MapPreset::Standard => MapSource::Standard,
            MapPreset::Wild => MapSource::Wild,
            MapPreset::Familiar => MapSource::Fresh,
        };
        Self {
            preset: Some(preset),
            source,
            generation: MapGenerationSettings {
                small_map: preset == MapPreset::Small,
                landmarks: match preset {
                    MapPreset::Familiar => vec![
                        Landmark::Ship,
                        Landmark::Kraid,
                        Landmark::Ridley,
                        Landmark::Phantoon,
                        Landmark::Draygon,
                        Landmark::MotherBrain,
                    ],
                    MapPreset::Standard => vec![Landmark::Ship, Landmark::MotherBrain],
                    _ => vec![],
                },
                maridia_water: climate,
                norfair_heat: climate,
                reward_missing_connect_utility: if preset == MapPreset::Wild { 0.0 } else { 2.0 },
                max_return_distance: if matches!(
                    preset,
                    MapPreset::Small | MapPreset::Familiar | MapPreset::Standard
                ) {
                    Some(13.0)
                } else {
                    None
                },
                reward_area_x: if preferred { 10.0 } else { 0.0 },
                reward_area_y: if preferred { 10.0 } else { 0.0 },
                target_area_rooms: if preset == MapPreset::Wild {
                    [1.0; 6]
                } else {
                    [150.0, 200.0, 200.0, 100.0, 200.0, 100.0]
                },
                target_area_x: if preferred {
                    [16.0, 10.0, 40.0, 48.0, 40.0, 16.0]
                } else {
                    [36.0; 6]
                },
                target_area_y: if preferred {
                    [12.0, 32.0, 64.0, 12.0, 32.0, 24.0]
                } else {
                    [36.0; 6]
                },
                reward_graph_diameter: 0.1,
                reward_save_distance: 0.1,
                reward_refill_distance: 0.1,
                reward_area_distinct_crossing: 1.0,
            },
            area_assignment: match source {
                MapSource::Small | MapSource::Standard | MapSource::Wild => {
                    Some(AreaAssignment::from_preset(AreaAssignmentPreset::Standard))
                }
                _ => None,
            },
        }
    }

    pub fn is_vanilla(&self) -> bool {
        self.source == MapSource::Vanilla
    }

    pub fn is_small(&self) -> bool {
        self.source == MapSource::Small
            || (self.source == MapSource::Fresh && self.generation.small_map)
    }

    pub fn label(&self) -> String {
        self.preset
            .map(|p| format!("{p:?}"))
            .unwrap_or_else(|| "Custom".to_string())
    }

    pub fn validate(&self, fresh_available: bool) -> Result<()> {
        if self.source == MapSource::Fresh {
            if !fresh_available {
                bail!("Fresh map generation is not configured on this server");
            }
            self.generation.validate()?;
        }
        Ok(())
    }
}
