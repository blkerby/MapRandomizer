mod logic_main;
mod logic_room;
mod logic_strat;
mod logic_tech;

use askama::Template;

#[derive(Template)]
#[template(path = "errors/room_not_found.html")]
struct RoomNotFoundTemplate {}

pub fn scope() -> actix_web::Scope {
    actix_web::web::scope("/logic")
        .service(logic_main::logic_main)
        .service(logic_room::logic_room)
        .service(logic_strat::logic_strat)
        .service(logic_tech::logic_tech)
}
