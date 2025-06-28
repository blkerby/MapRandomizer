use crate::web::{AppData, VersionInfo};
use actix_web::{get, web, HttpResponse, Responder};
use askama::Template;
use hashbrown::HashMap;

#[derive(Template)]
#[template(path = "about.html")]
struct AboutTemplate {
    version_info: VersionInfo,
    sprite_artists: Vec<String>,
    video_creators: Vec<(String, usize)>,
}

impl AboutTemplate {
    fn sprite_artists(&self) -> String {
        self.sprite_artists
            .iter()
            .map(|x| format!("<i>{x}</i>"))
            .collect::<Vec<String>>()
            .join(", ")
    }

    fn video_creators(&self) -> String {
        self.video_creators
            .iter()
            .map(|x| format!("<i>{}</i>", x.0))
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

    let mut video_creator_cnt: HashMap<String, usize> = HashMap::new();
    for (_, video_list) in app_data.game_data.strat_videos.iter() {
        for video in video_list {
            *video_creator_cnt
                .entry(video.created_user.clone())
                .or_default() += 1;
        }
    }
    let mut video_creators: Vec<(String, usize)> = video_creator_cnt.into_iter().collect();
    video_creators.sort_by_key(|x| x.1);
    video_creators.reverse();

    let about_template = AboutTemplate {
        version_info: app_data.version_info.clone(),
        sprite_artists,
        video_creators,
    };
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(about_template.render().unwrap())
}
