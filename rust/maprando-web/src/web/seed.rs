mod customize_seed;
mod get_seed_file;
mod unlock_seed;
mod view_seed;

pub fn scope() -> actix_web::Scope {
    actix_web::web::scope("/seed")
        .service(view_seed::view_seed)
        .service(get_seed_file::get_seed_file)
        .service(customize_seed::customize_seed)
        .service(unlock_seed::unlock_seed)
        .service(view_seed::view_seed_redirect)
}
