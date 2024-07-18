use crate::{web::logic::RoomNotFoundTemplate, web::AppData};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;

#[get("/room/{room_name}/{from_node}/{to_node}/{strat_name}")]
async fn logic_strat(
    info: web::Path<(String, usize, usize, String)>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let room_name = &info.0;
    let from_node = info.1;
    let to_node = info.2;
    let strat_name = &info.3;
    if let Some(html) = app_data.logic_data.strat_html.get(&(
        room_name.clone(),
        from_node,
        to_node,
        strat_name.clone(),
    )) {
        HttpResponse::Ok().body(html.clone())
    } else {
        let template = RoomNotFoundTemplate {};
        HttpResponse::NotFound().body(template.render().unwrap())
    }
}
