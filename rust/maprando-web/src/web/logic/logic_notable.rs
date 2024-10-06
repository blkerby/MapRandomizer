use crate::{web::logic::RoomNotFoundTemplate, web::AppData};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;
use maprando_game::{NotableId, RoomId};

#[get("/notable/{room_id}/{notable_id}")]
async fn logic_notable(info: web::Path<(RoomId, NotableId)>, app_data: web::Data<AppData>) -> impl Responder {
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
