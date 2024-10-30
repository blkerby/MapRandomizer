use crate::web::{AppData, VersionInfo};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;
use maprando::settings::SkillAssumptionSettings;

#[derive(Template)]
#[template(path = "logic/boss_calculator.html")]
struct BossCalculatorTemplate<'a> {
    version_info: VersionInfo,
    presets: &'a [SkillAssumptionSettings],
    presets_json: String,
}

#[get("/boss_calculator")]
async fn logic_boss_calculator(app_data: web::Data<AppData>) -> impl Responder {
    let mut presets = vec![];
    for p in &app_data.preset_data.skill_presets {
        let preset_name = p.preset.as_ref().unwrap();
        if preset_name == "Implicit" || preset_name == "Beyond" || preset_name == "Ignored" {
            continue;
        }
        presets.push(p.clone());
    }

    let template = BossCalculatorTemplate {
        version_info: app_data.version_info.clone(),
        presets: &presets,
        presets_json: serde_json::to_string(&presets).unwrap(),
    };
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(template.render().unwrap())
}
