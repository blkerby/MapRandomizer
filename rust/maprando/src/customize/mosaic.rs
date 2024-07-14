use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct MosaicTheme {
    pub name: String,
    pub display_name: String,
}
