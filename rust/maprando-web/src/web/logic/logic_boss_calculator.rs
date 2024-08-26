use crate::web::{AppData, VersionInfo};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;

#[derive(Template)]
#[template(path = "logic/boss_calculator.html")]
struct BossCalculatorTemplate<'a> {
    version_info: VersionInfo,
    presets_json: String,
    difficulty_names: Vec<&'a String>,
    boss_proficiencies: Vec<(f32, f32, f32, f32, f32)>,
}

#[get("/boss_calculator")]
async fn logic_boss_calculator(app_data: web::Data<AppData>) -> impl Responder {
    let mut difficulty_names: Vec<&String> = app_data
        .preset_data
        .iter()
        .map(|preset| &preset.preset.name)
        .collect();

    // Skip Beyond
    difficulty_names.pop();

    let boss_proficiencies: Vec<(f32, f32, f32, f32, f32)> = app_data
        .preset_data
        .iter()
        .map(|preset| {
            (
                preset.preset.phantoon_proficiency,
                preset.preset.draygon_proficiency,
                preset.preset.ridley_proficiency,
                preset.preset.botwoon_proficiency,
                preset.preset.mother_brain_proficiency,
            )
        })
        .collect();

    let template = BossCalculatorTemplate {
        version_info: app_data.version_info.clone(),
        presets_json: serde_json::to_string(&app_data.preset_data).unwrap(),
        difficulty_names,
        boss_proficiencies,
    };
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(template.render().unwrap())
}
