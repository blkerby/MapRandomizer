mod logic_boss_calculator;
mod logic_main;
mod logic_notable;
mod logic_room;
mod logic_strat;
mod logic_tech;
mod logic_vanilla_map;

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
        .service(logic_notable::logic_notable)
        .service(logic_boss_calculator::logic_boss_calculator)
        .service(logic_vanilla_map::logic_vanilla_map)
}
