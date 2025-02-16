use std::path::Path;

use anyhow::{Context, Result};
use hashbrown::HashMap;
use log::warn;
use maprando_game::{GameData, IndexedVec, NotableId, RoomId, TechId};
use serde::{Deserialize, Serialize};

use crate::{
    randomize::DifficultyConfig,
    settings::{
        ItemProgressionSettings, ObjectiveSettings, QualityOfLifeSettings, RandomizerSettings, SkillAssumptionSettings
    },
};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TechData {
    pub tech_id: TechId,
    pub name: String,
    pub difficulty: String,
    pub video_id: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NotableData {
    pub room_id: usize,
    pub notable_id: usize,
    pub room_name: String,
    pub name: String,
    pub difficulty: String,
    pub video_id: Option<usize>,
}

pub struct PresetData {
    pub tech_data_map: HashMap<TechId, TechData>,
    pub notable_data_map: HashMap<(RoomId, NotableId), NotableData>,
    pub difficulty_levels: IndexedVec<String>,
    pub tech_by_difficulty: HashMap<String, Vec<TechId>>,
    pub notables_by_difficulty: HashMap<String, Vec<(RoomId, NotableId)>>,
    pub skill_presets: Vec<SkillAssumptionSettings>,
    pub item_progression_presets: Vec<ItemProgressionSettings>,
    pub quality_of_life_presets: Vec<QualityOfLifeSettings>,
    pub objective_presets: Vec<ObjectiveSettings>,
    pub difficulty_tiers: Vec<DifficultyConfig>,
    pub full_presets: Vec<RandomizerSettings>,
    pub default_preset: RandomizerSettings,
}

fn get_tech_by_difficulty(
    tech_data: &Vec<TechData>,
    difficulty_levels: &[String],
) -> HashMap<String, Vec<TechId>> {
    let mut out: HashMap<String, Vec<TechId>> = HashMap::new();
    for d in difficulty_levels {
        out.insert(d.clone(), vec![]);
    }
    for data in tech_data {
        out.get_mut(&data.difficulty).unwrap().push(data.tech_id);
    }
    out
}

fn get_notables_by_difficulty(
    notable_data: &[NotableData],
    difficulty_levels: &[String],
) -> HashMap<String, Vec<(RoomId, NotableId)>> {
    let mut out: HashMap<String, Vec<(RoomId, NotableId)>> = HashMap::new();
    for d in difficulty_levels {
        out.insert(d.clone(), vec![]);
    }
    for data in notable_data {
        out.get_mut(&data.difficulty)
            .unwrap()
            .push((data.room_id, data.notable_id));
    }
    out
}

impl PresetData {
    pub fn load(
        tech_path: &Path,
        notable_path: &Path,
        presets_path: &Path,
        game_data: &GameData,
    ) -> Result<Self> {
        let tech_data_str = std::fs::read_to_string(tech_path)
            .context(format!("reading from {}", tech_path.display()))?;
        let mut tech_data: Vec<TechData> = serde_json::from_str(&tech_data_str)?;
        for d in &mut tech_data {
            if d.difficulty == "Uncategorized" {
                warn!("Uncategorized tech ({}): {}", d.tech_id, d.name);
                d.difficulty = "Ignored".to_string();
            }
        }
        let tech_data_map: HashMap<TechId, TechData> = tech_data
            .clone()
            .into_iter()
            .map(|x| (x.tech_id, x))
            .collect();

        let notable_data_str = std::fs::read_to_string(notable_path)
            .context(format!("reading from {}", notable_path.display()))?;
        let mut notable_data: Vec<NotableData> = serde_json::from_str(&notable_data_str)?;
        for d in &mut notable_data {
            if d.difficulty == "Uncategorized" {
                warn!(
                    "Uncategorized notable ({}, {}) {}: ({})",
                    d.room_id, d.notable_id, d.room_name, d.name
                );
                d.difficulty = "Ignored".to_string();
            }
        }
        let notable_data_map: HashMap<(RoomId, NotableId), NotableData> = notable_data
            .clone()
            .into_iter()
            .map(|x| ((x.room_id, x.notable_id), x))
            .collect();

        let mut difficulty_levels: IndexedVec<String> = IndexedVec::default();
        for d in [
            "Implicit",
            "Basic",
            "Medium",
            "Hard",
            "Very Hard",
            "Expert",
            "Extreme",
            "Insane",
            "Beyond",
            "Ignored",
        ] {
            difficulty_levels.add(&d.to_string());
        }

        let tech_by_difficulty = get_tech_by_difficulty(&tech_data, &difficulty_levels.keys);
        let notables_by_difficulty =
            get_notables_by_difficulty(&notable_data, &difficulty_levels.keys);

        let implicit_tech = &tech_by_difficulty["Implicit"];
        let implicit_notables = &notables_by_difficulty["Implicit"];

        let skill_preset_names = difficulty_levels.keys[..9].to_vec();
        let skill_preset_path = presets_path.join("skill-assumptions");
        let mut skill_presets: Vec<SkillAssumptionSettings> = vec![];
        let mut difficulty_tiers: Vec<DifficultyConfig> = vec![];
        for name in skill_preset_names {
            let path = skill_preset_path.join(format!("{}.json", name));
            let preset_str = std::fs::read_to_string(path.clone())
                .context(format!("reading from {}", path.display()))?;
            let preset: SkillAssumptionSettings =
                serde_json::from_str(&preset_str).context(format!("parsing {}", path.display()))?;
            let difficulty =
                DifficultyConfig::new(&preset, game_data, implicit_tech, implicit_notables);
            assert!(preset.preset == Some(name.to_string()));
            skill_presets.push(preset);
            if name != "Implicit" && name != "Ignored" {
                difficulty_tiers.push(difficulty);
            }
        }
        difficulty_tiers.reverse();

        let item_progression_preset_names =
            ["Normal", "Tricky", "Technical", "Challenge", "Desolate"];
        let item_progression_preset_path = presets_path.join("item-progression");
        let mut item_progression_presets: Vec<ItemProgressionSettings> = vec![];
        for name in item_progression_preset_names {
            let path = item_progression_preset_path.join(format!("{}.json", name));
            let preset_str = std::fs::read_to_string(path.clone())
                .context(format!("reading from {}", path.display()))?;
            let preset: ItemProgressionSettings =
                serde_json::from_str(&preset_str).context(format!("parsing {}", path.display()))?;
            assert!(preset.preset == Some(name.to_string()));
            item_progression_presets.push(preset);
        }

        let qol_preset_names = ["Off", "Low", "Default", "High", "Max"];
        let qol_preset_path = presets_path.join("quality-of-life");
        let mut quality_of_life_presets: Vec<QualityOfLifeSettings> = vec![];
        for name in qol_preset_names {
            let path = qol_preset_path.join(format!("{}.json", name));
            let preset_str = std::fs::read_to_string(path.clone())
                .context(format!("reading from {}", path.display()))?;
            let preset: QualityOfLifeSettings =
                serde_json::from_str(&preset_str).context(format!("parsing {}", path.display()))?;
            assert!(preset.preset == Some(name.to_string()));
            quality_of_life_presets.push(preset);
        }

        let objective_preset_names = ["None", "Bosses", "Minibosses", "Chozos", "Pirates", "Metroids", "Random"];
        let objective_preset_path = presets_path.join("objectives");
        let mut objective_presets: Vec<ObjectiveSettings> = vec![];
        for name in objective_preset_names {
            let path = objective_preset_path.join(format!("{}.json", name));
            let preset_str = std::fs::read_to_string(path.clone())
                .context(format!("reading from {}", path.display()))?;
            let preset: ObjectiveSettings =
                serde_json::from_str(&preset_str).context(format!("parsing {}", path.display()))?;
            assert!(preset.preset == Some(name.to_string()));
            objective_presets.push(preset);
        }

        let full_preset_names = [
            "Default",
            "Community Race Season 3 (No animals)",
            "Community Race Season 3 (Save the animals)",
        ];
        let full_preset_path = presets_path.join("full-settings");
        let mut full_presets: Vec<RandomizerSettings> = vec![];
        for name in full_preset_names {
            let path = full_preset_path.join(format!("{}.json", name));
            let preset_str = std::fs::read_to_string(path.clone())
                .context(format!("reading from {}", path.display()))?;
            let preset: RandomizerSettings =
                serde_json::from_str(&preset_str).context(format!("parsing {}", path.display()))?;
            full_presets.push(preset);
        }

        let preset_data = Self {
            tech_data_map,
            notable_data_map,
            difficulty_levels,
            tech_by_difficulty,
            notables_by_difficulty,
            skill_presets,
            item_progression_presets,
            quality_of_life_presets,
            objective_presets,
            difficulty_tiers,
            default_preset: full_presets[0].clone(),
            full_presets,
        };
        Ok(preset_data)
    }
}
