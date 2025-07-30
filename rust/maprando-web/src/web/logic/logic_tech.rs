use crate::{web::AppData, web::logic::RoomNotFoundTemplate};
use actix_web::{HttpResponse, Responder, get, web};
use askama::Template;
use maprando_game::TechId;

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
