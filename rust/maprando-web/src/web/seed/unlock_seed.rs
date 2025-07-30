use crate::web::AppData;
use actix_web::{
    HttpResponse, Responder,
    http::header::{self},
    post, web,
};
use askama::Template;
use serde_derive::Deserialize;
use std::time::SystemTime;

#[derive(Template)]
#[template(path = "errors/invalid_token.html")]
struct InvalidTokenTemplate {}

#[derive(Template)]
#[template(path = "errors/already_unlocked.html")]
struct AlreadyUnlockedTemplate {}

#[derive(Deserialize)]
struct UnlockRequest {
    spoiler_token: String,
}

#[post("/{name}/unlock")]
async fn unlock_seed(
    req: web::Form<UnlockRequest>,
    info: web::Path<(String,)>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let seed_name = &info.0;
    let seed_spoiler_token = app_data
        .seed_repository
        .get_file(seed_name, "spoiler_token.txt")
        .await
        .unwrap();

    if req.spoiler_token.as_bytes() == seed_spoiler_token {
        let unlocked_timestamp_data = app_data
            .seed_repository
            .get_file(seed_name, "unlocked_timestamp.txt")
            .await;
        if unlocked_timestamp_data.is_ok() {
            // TODO: handle other errors that are not 404.
            let template = AlreadyUnlockedTemplate {};
            return HttpResponse::UnprocessableEntity().body(template.render().unwrap());
        }

        app_data
            .seed_repository
            .move_prefix(seed_name, "locked", "public")
            .await
            .unwrap();
        let timestamp = match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            Ok(n) => n.as_millis() as usize,
            Err(_) => panic!("SystemTime before UNIX EPOCH!"),
        };
        let unlock_time_str = format!("{timestamp}");
        app_data
            .seed_repository
            .put_file(
                seed_name,
                "unlocked_timestamp.txt".to_string(),
                unlock_time_str.into_bytes(),
            )
            .await
            .unwrap();
    } else {
        let template = InvalidTokenTemplate {};
        return HttpResponse::Forbidden().body(template.render().unwrap());
    }
    HttpResponse::Found()
        .insert_header((header::LOCATION, format!("/seed/{}/", info.0)))
        .finish()
}
