use crate::{web::logic::RoomNotFoundTemplate, web::AppData};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;

#[get("/room/{room_id}/{from_node}/{to_node}/{strat_name}")]
async fn logic_strat(
    info: web::Path<(usize, usize, usize, usize)>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let room_id = info.0;
    let from_node = info.1;
    let to_node = info.2;
    let strat_id = info.3;
    if let Some(html) = app_data.logic_data.strat_html.get(&(
        room_id,
        from_node,
        to_node,
        strat_id,
    )) {
        HttpResponse::Ok()
            .content_type("text/html; charset=utf-8")
            .body(html.clone())
    } else {
        let template = RoomNotFoundTemplate {};
        HttpResponse::NotFound().body(template.render().unwrap())
    }
}
