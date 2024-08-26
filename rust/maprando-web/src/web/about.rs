use crate::web::{AppData, VersionInfo};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;

#[derive(Template)]
#[template(path = "about.html")]
struct AboutTemplate {
    version_info: VersionInfo,
    sprite_artists: Vec<String>,
}

impl AboutTemplate {
    fn sprite_artists(&self) -> String {
        self.sprite_artists
            .iter()
            .map(|x| format!("<i>{}</i>", x))
            .collect::<Vec<String>>()
            .join(", ")
    }
}

#[get("/about")]
async fn about(app_data: web::Data<AppData>) -> impl Responder {
    let mut sprite_artists = vec![];

    for category in &app_data.samus_sprite_categories {
        for info in &category.sprites {
            for author in &info.authors {
                if info.display_name != "Samus" {
                    sprite_artists.push(author.clone());
                }
            }
        }
    }
    sprite_artists.sort_by_key(|x| x.to_lowercase());
    sprite_artists.dedup();
    let about_template = AboutTemplate {
        version_info: app_data.version_info.clone(),
        sprite_artists,
    };
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(about_template.render().unwrap())
}
