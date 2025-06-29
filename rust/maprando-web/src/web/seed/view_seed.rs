use crate::web::{AppData, VersionInfo};
use actix_web::{
    get,
    http::header::{self, CacheControl, CacheDirective},
    web, HttpResponse, Responder,
};
use askama::Template;
use log::error;
use maprando::customize::{mosaic::MosaicTheme, samus_sprite::SamusSpriteCategory};

#[derive(Template)]
#[template(path = "errors/seed_not_found.html")]
struct SeedNotFoundTemplate {}

#[derive(Template)]
#[template(path = "seed/customize_seed.html")]
struct CustomizeSeedTemplate {
    version_info: VersionInfo,
    spoiler_token_prefix: String,
    unlocked_timestamp_str: String,
    seed_header: String,
    seed_footer: String,
    samus_sprite_categories: Vec<SamusSpriteCategory>,
    etank_colors: Vec<Vec<String>>,
    mosaic_themes: Vec<MosaicTheme>,
}

#[get("/{name}")]
async fn view_seed_redirect(info: web::Path<(String,)>) -> impl Responder {
    // Redirect to the seed page (with the trailing slash):
    HttpResponse::Found()
        .insert_header((header::LOCATION, format!("{}/", info.0)))
        .finish()
}

#[get("/{name}/")]
async fn view_seed(info: web::Path<(String,)>, app_data: web::Data<AppData>) -> impl Responder {
    let seed_name = &info.0;
    let (seed_header, seed_footer, unlocked_timestamp_str, spoiler_token) = futures::join!(
        app_data
            .seed_repository
            .get_file(seed_name, "seed_header.html"),
        app_data
            .seed_repository
            .get_file(seed_name, "seed_footer.html"),
        app_data
            .seed_repository
            .get_file(seed_name, "unlocked_timestamp.txt"),
        app_data
            .seed_repository
            .get_file(seed_name, "spoiler_token.txt"),
    );
    let spoiler_token = String::from_utf8(spoiler_token.unwrap_or(vec![])).unwrap();
    let spoiler_token_prefix = if spoiler_token.is_empty() {
        "".to_string()
    } else {
        spoiler_token[0..16].to_string()
    };
    match (seed_header, seed_footer) {
        (Ok(header), Ok(footer)) => {
            let customize_template = CustomizeSeedTemplate {
                version_info: app_data.version_info.clone(),
                unlocked_timestamp_str: String::from_utf8(unlocked_timestamp_str.unwrap_or(vec![]))
                    .unwrap(),
                spoiler_token_prefix: spoiler_token_prefix.to_string(),
                seed_header: String::from_utf8(header.to_vec()).unwrap(),
                seed_footer: String::from_utf8(footer.to_vec()).unwrap(),
                samus_sprite_categories: app_data.samus_sprite_categories.clone(),
                etank_colors: app_data.etank_colors.clone(),
                mosaic_themes: app_data.mosaic_themes.clone(),
            };
            // We use a no-cache directive to prevent problems when we change the JavaScript.
            // Probably better would be to properly version the JS and control cache that way.
            HttpResponse::Ok()
                .content_type("text/html; charset=utf-8")
                .insert_header(CacheControl(vec![CacheDirective::NoCache]))
                .body(customize_template.render().unwrap())
        }
        (Err(err), _) => {
            error!("{err}");
            let template = SeedNotFoundTemplate {};
            HttpResponse::NotFound().body(template.render().unwrap())
        }
        (_, Err(err)) => {
            error!("{err}");
            let template = SeedNotFoundTemplate {};
            HttpResponse::NotFound().body(template.render().unwrap())
        }
    }
}
