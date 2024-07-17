use crate::{web::logic::RoomNotFoundTemplate, web::AppData};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;

#[get("/room/{name}")]
async fn logic_room(info: web::Path<(String,)>, app_data: web::Data<AppData>) -> impl Responder {
    let room_name = &info.0;
    if let Some(html) = app_data.logic_data.room_html.get(room_name) {
        HttpResponse::Ok().body(html.clone())
    } else {
        let template = RoomNotFoundTemplate {};
        HttpResponse::NotFound().body(template.render().unwrap())
    }
}
