use crate::web::{AppData, VersionInfo};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;

#[derive(Template)]
#[template(path = "releases.html")]
struct ReleasesTemplate {
    version_info: VersionInfo,
}

#[get("/releases")]
async fn releases(app_data: web::Data<AppData>) -> impl Responder {
    let changes_template = ReleasesTemplate {
        version_info: app_data.version_info.clone(),
    };
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(changes_template.render().unwrap())
}
