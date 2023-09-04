use serde::{de::Error, Deserialize, Deserializer};

fn from_hex<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    usize::from_str_radix(&s, 16).map_err(D::Error::custom)
}

fn from_hex_words<'de, D>(deserializer: D) -> Result<Vec<u16>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    let mut out: Vec<u16> = vec![];
    for word in s.split_ascii_whitespace() {
        out.push(u16::from_str_radix(word, 16).map_err(D::Error::custom)?);
    }
    Ok(out)
}

#[derive(Debug, Deserialize)]
pub struct FX1 {
    #[serde(deserialize_with="from_hex")]
    pub surfacestart: usize,
    #[serde(deserialize_with="from_hex")]
    pub surfacenew: usize,
    #[serde(deserialize_with="from_hex")]
    pub surfacespeed: usize,
    #[serde(deserialize_with="from_hex")]
    pub surfacedelay: usize,
    #[serde(rename="type", deserialize_with="from_hex")]
    pub type_: usize,
    #[serde(rename="transparency1_A", deserialize_with="from_hex")]
    pub transparency1_a: usize,
    #[serde(rename="transparency2_B", deserialize_with="from_hex")]
    pub transparency2_b: usize,
    #[serde(rename="liquidflags_C", deserialize_with="from_hex")]
    pub liquidflags_c: usize,
    #[serde(deserialize_with="from_hex")]
    pub paletteflags: usize,
    #[serde(deserialize_with="from_hex")]
    pub animationflags: usize,
    #[serde(deserialize_with="from_hex")]
    pub paletteblend: usize,
}

#[derive(Debug, Deserialize)]
pub struct FX1List {
    #[serde(rename="FX1", default)]
    pub fx1: Vec<FX1>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
pub enum Layer2Type {
    Layer2,
    BGData,
}

#[derive(Debug, Deserialize)]
pub struct Screen {
    #[serde(rename="X", deserialize_with="from_hex")]
    pub x: usize,
    #[serde(rename="Y", deserialize_with="from_hex")]
    pub y: usize,
    #[serde(rename="$value", deserialize_with="from_hex_words")]
    pub data: Vec<u16>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Layer1 {
    #[serde(rename="Screen")]
    pub screen: Vec<Screen>
}

#[derive(Debug, Deserialize, Default)]
pub struct Layer2 {
    #[serde(rename="Screen")]
    pub screen: Vec<Screen>
}

#[derive(Debug, Deserialize)]
pub struct BTS {
    #[serde(rename="Screen")]
    pub screen: Vec<Screen>
}

#[derive(Debug, Deserialize)]
pub struct LevelData {
    #[serde(rename="Width", deserialize_with="from_hex")]
    pub width: usize,
    #[serde(rename="Height", deserialize_with="from_hex")]
    pub height: usize,
    #[serde(rename="Layer1")]
    pub layer_1: Layer1,
    #[serde(rename="BTS")]
    pub bts: BTS,
    #[serde(rename="Layer2", default)]
    pub layer_2: Layer2,
}

#[derive(Debug, Deserialize, Default)]
pub struct BGDataData {
    #[serde(rename="Type", default)]
    pub type_: String,
    #[serde(rename="SOURCE", default)]
    pub source: String,
    #[serde(rename="DEST", default)]
    pub dest: String,
    #[serde(rename="SIZE", default)]
    pub size: String,
}

#[derive(Debug, Deserialize)]
pub struct BGData {
    #[serde(rename="Data", default)]
    pub data: Vec<BGDataData>,
}

#[derive(Debug, Deserialize)]
pub struct RoomState {
    pub condition: String,
    #[serde(rename="Arg", deserialize_with="from_hex", default)]
    pub arg: usize,
    #[serde(rename="GFXset", deserialize_with="from_hex")]
    pub gfx_set: usize,
    #[serde(rename="FX1s")]
    pub fx1s: FX1List,
    #[serde(rename="LevelData")]
    pub level_data: LevelData,
    pub layer2_type: Layer2Type,
    #[serde(deserialize_with="from_hex")]
    pub layer2_xscroll: usize,
    #[serde(deserialize_with="from_hex")]
    pub layer2_yscroll: usize,
    #[serde(rename="BGData")]
    pub bg_data: BGData,
}

#[derive(Debug, Deserialize)]
pub struct RoomStateList {
    #[serde(rename="State")]
    pub state: Vec<RoomState>,
}

#[derive(Debug, Deserialize)]
pub struct Room {
    #[serde(deserialize_with="from_hex")]
    pub area: usize,
    #[serde(deserialize_with="from_hex")]
    pub index: usize,
    #[serde(rename="specialGFX", deserialize_with="from_hex")]
    pub special_gfx: usize,
    #[serde(rename="States")]
    pub states: RoomStateList,
}
