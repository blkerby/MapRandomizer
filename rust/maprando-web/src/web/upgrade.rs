use crate::web::AppData;
use actix_web::{HttpResponse, Responder, post, web};
use log::error;
use maprando::settings::try_upgrade_settings;

#[post("/upgrade-settings")]
async fn upgrade_settings(settings_str: String, app_data: web::Data<AppData>) -> impl Responder {
    match try_upgrade_settings(settings_str, &app_data.preset_data, true) {
        Ok((settings_str, _)) => HttpResponse::Ok()
            .content_type("application/json")
            .body(settings_str),
        Err(e) => {
            error!("Failed to upgrade settings: {e}");
            HttpResponse::BadRequest().body(e.to_string())
        }
    }
}
