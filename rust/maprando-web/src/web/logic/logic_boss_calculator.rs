use crate::web::{AppData, VersionInfo};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;

#[derive(Template)]
#[template(path = "logic/boss_calculator.html")]
struct BossCalculatorTemplate {
    version_info: VersionInfo,
    presets_json: String,
}

#[get("/boss_calculator")]
async fn logic_boss_calculator(app_data: web::Data<AppData>) -> impl Responder {
    let template = BossCalculatorTemplate {
        version_info: app_data.version_info.clone(),
        presets_json: serde_json::to_string(&app_data.preset_data).unwrap(),
    };
    HttpResponse::Ok().body(template.render().unwrap())
}
