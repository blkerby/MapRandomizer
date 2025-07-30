use crate::{web::AppData, web::logic::RoomNotFoundTemplate};
use actix_web::{HttpResponse, Responder, get, web};
use askama::Template;

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
