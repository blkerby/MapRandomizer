use crate::web::AppData;
use actix_web::{HttpResponse, Responder, get, web};

#[get("/vanilla_map.png")]
async fn logic_vanilla_map(app_data: web::Data<AppData>) -> impl Responder {
    HttpResponse::Ok()
        .content_type("image/png")
        .body(app_data.logic_data.vanilla_map_png.clone())
}
