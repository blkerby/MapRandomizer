use actix_web::{HttpResponse, Responder, get, web};
use askama::Template;
use maprando::settings::SkillAssumptionSettings;
use maprando_game::{NotableId, RoomId, TechId};

use crate::web::{AppData, VersionInfo};

#[derive(Template)]
#[template(path = "errors/room_not_found.html")]
struct RoomNotFoundTemplate {}

#[get("")]
async fn logic_main(app_data: web::Data<AppData>) -> impl Responder {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(app_data.logic_data.index_html.clone())
}

#[get("/room/{name}")]
async fn logic_room(info: web::Path<(usize,)>, app_data: web::Data<AppData>) -> impl Responder {
    let room_id = &info.0;
    if let Some(html) = app_data.logic_data.room_html.get(room_id) {
        HttpResponse::Ok()
            .content_type("text/html; charset=utf-8")
            .body(html.clone())
    } else {
        let template = RoomNotFoundTemplate {};
        HttpResponse::NotFound().body(template.render().unwrap())
    }
}

#[get("/room/{room_id}/{from_node}/{to_node}/{strat_name}")]
async fn logic_strat(
    info: web::Path<(usize, usize, usize, usize)>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let room_id = info.0;
    let from_node = info.1;
    let to_node = info.2;
    let strat_id = info.3;
    if let Some(html) = app_data
        .logic_data
        .strat_html
        .get(&(room_id, from_node, to_node, strat_id))
    {
        HttpResponse::Ok()
            .content_type("text/html; charset=utf-8")
            .body(html.clone())
    } else {
        let template = RoomNotFoundTemplate {};
        HttpResponse::NotFound().body(template.render().unwrap())
    }
}

#[get("/tech/{tech_id}")]
async fn logic_tech(info: web::Path<(TechId,)>, app_data: web::Data<AppData>) -> impl Responder {
    let tech_id = info.0;
    if let Some(html) = app_data.logic_data.tech_html.get(&tech_id) {
        HttpResponse::Ok()
            .content_type("text/html; charset=utf-8")
            .body(html.clone())
    } else {
        let template = RoomNotFoundTemplate {};
        HttpResponse::NotFound().body(template.render().unwrap())
    }
}

#[get("/notable/{room_id}/{notable_id}")]
async fn logic_notable(
    info: web::Path<(RoomId, NotableId)>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let room_id = info.0;
    let notable_id = info.1;
    if let Some(html) = app_data.logic_data.notable_html.get(&(room_id, notable_id)) {
        HttpResponse::Ok()
            .content_type("text/html; charset=utf-8")
            .body(html.clone())
    } else {
        let template = RoomNotFoundTemplate {};
        HttpResponse::NotFound().body(template.render().unwrap())
    }
}

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

#[get("/vanilla_map.png")]
async fn logic_vanilla_map(app_data: web::Data<AppData>) -> impl Responder {
    HttpResponse::Ok()
        .content_type("image/png")
        .body(app_data.logic_data.vanilla_map_png.clone())
}

pub fn scope() -> actix_web::Scope {
    actix_web::web::scope("/logic")
        .service(logic_main)
        .service(logic_room)
        .service(logic_strat)
        .service(logic_tech)
        .service(logic_notable)
        .service(logic_boss_calculator)
        .service(logic_vanilla_map)
}
