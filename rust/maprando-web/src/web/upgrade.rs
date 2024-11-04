use crate::web::AppData;
use actix_web::{post, web, HttpResponse, Responder};

#[post("/upgrade-settings")]
async fn upgrade_settings(
    settings_str: String,
    _app_data: web::Data<AppData>,
) -> actix_web::Result<impl Responder> {
    let settings: serde_json::Value = serde_json::from_str(&settings_str)?;
    let out = serde_json::to_string(&settings)?;
    Ok(HttpResponse::Ok()
        .content_type("application/json")
        .body(out))
}
