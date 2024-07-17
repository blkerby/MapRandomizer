use crate::web::{AppData, VersionInfo};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;

#[derive(Template)]
#[template(path = "home.html")]
struct HomeTemplate {
    version_info: VersionInfo,
}

#[get("/")]
async fn home(app_data: web::Data<AppData>) -> impl Responder {
    let home_template = HomeTemplate {
        version_info: app_data.version_info.clone(),
    };
    HttpResponse::Ok().body(home_template.render().unwrap())
}
