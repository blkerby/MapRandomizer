use crate::{web::AppData, VISUALIZER_PATH};
use actix_web::{get, web, HttpResponse, Responder};
use anyhow::{Context, Result};
use askama::Template;
use log::error;
use std::path::Path;

#[derive(Template)]
#[template(path = "errors/file_not_found.html")]
struct FileNotFoundTemplate {}

#[get("/{name}/data/{filename:.*}")]
async fn get_seed_file(
    info: web::Path<(String, String)>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let seed_name = &info.0;
    let filename = &info.1;
    println!("get_seed_file {filename}");

    let data_result: Result<Vec<u8>> = if filename.starts_with("visualizer/")
        && app_data.static_visualizer
    {
        let path = Path::new(VISUALIZER_PATH).join(filename.strip_prefix("visualizer/").unwrap());
        std::fs::read(&path)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("Error reading static file: {}", path.display()))
    } else {
        app_data
            .seed_repository
            .get_file(seed_name, &("public/".to_string() + filename))
            .await
    };

    match data_result {
        Ok(data) => {
            let ext = Path::new(filename)
                .extension()
                .map(|x| x.to_str().unwrap())
                .unwrap_or("bin");
            let mime = actix_files::file_extension_to_mime(ext);
            HttpResponse::Ok().content_type(mime).body(data)
        }
        // TODO: Use more refined error handling instead of always returning 404:
        Err(err) => {
            error!("{err}");
            HttpResponse::NotFound().body(FileNotFoundTemplate {}.render().unwrap())
        }
    }
}
