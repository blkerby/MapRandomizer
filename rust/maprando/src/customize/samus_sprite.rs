use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct SamusSpriteInfo {
    pub name: String,
    pub display_name: String,
    pub credits_name: Option<String>,
    pub authors: Vec<String>,
}

#[derive(Deserialize, Clone)]
pub struct SamusSpriteCategory {
    pub category_name: String,
    pub sprites: Vec<SamusSpriteInfo>,
}
