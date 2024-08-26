use crate::web::AppData;
use actix_web::{get, web, HttpResponse, Responder};

#[get("")]
async fn logic_main(app_data: web::Data<AppData>) -> impl Responder {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(app_data.logic_data.index_html.clone())
}
