use crate::{web::logic::RoomNotFoundTemplate, web::AppData};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;
use maprando_game::TechId;

#[get("/tech/{name}")]
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
