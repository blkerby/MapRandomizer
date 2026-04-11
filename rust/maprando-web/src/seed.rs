use std::path::Path;

use crate::{
    VISUALIZER_PATH,
    web::{AppData, VersionInfo},
};
use actix_easy_multipart::{MultipartForm, bytes::Bytes, text::Text};
use actix_web::{
    HttpResponse, Responder, get,
    http::header::{
        self, CacheControl, CacheDirective, ContentDisposition, DispositionParam, DispositionType,
    },
    post, web,
};
use anyhow::{Context, Result};
use askama::Template;
use log::error;
use log::info;
use serde_derive::Deserialize;
use std::time::SystemTime;

use maprando::customize::{
    StatuesHallwayAudio, StatuesHallwayTiling, mosaic::MosaicTheme,
    samus_sprite::SamusSpriteCategory,
};
use maprando::{
    customize::{
        ControllerButton, ControllerConfig, CustomizeSettings, DoorTheme, FlashingSetting,
        MusicSettings, PaletteTheme, ShakingSetting, TileTheme, parse_controller_button,
    },
    patch::{Rom, make_rom},
    randomize::Randomization,
    settings::{RandomizerSettings, try_upgrade_settings},
};

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

#[derive(Template)]
#[template(path = "errors/invalid_rom.html")]
struct InvalidRomTemplate {}

#[derive(MultipartForm)]
struct CustomizeRequest {
    rom: Bytes,
    samus_sprite: Text<String>,
    etank_color: Text<String>,
    item_dot_change: Text<String>,
    transition_letters: Text<bool>,
    reserve_hud_style: Text<bool>,
    room_palettes: Text<String>,
    tile_theme: Text<String>,
    door_theme: Text<String>,
    music: Text<String>,
    disable_beeping: Text<bool>,
    shaking: Text<String>,
    flashing: Text<String>,
    vanilla_screw_attack_animation: Text<bool>,
    save_icons: Text<bool>,
    boss_icons: Text<bool>,
    miniboss_icons: Text<bool>,
    room_names: Text<bool>,
    statues_hallway_tiling: Text<String>,
    statues_hallway_audio: Text<String>,
    control_shot: Text<String>,
    control_jump: Text<String>,
    control_dash: Text<String>,
    control_item_select: Text<String>,
    control_item_cancel: Text<String>,
    control_angle_up: Text<String>,
    control_angle_down: Text<String>,
    spin_lock_left: Option<Text<String>>,
    spin_lock_right: Option<Text<String>>,
    spin_lock_up: Option<Text<String>>,
    spin_lock_down: Option<Text<String>>,
    spin_lock_x: Option<Text<String>>,
    spin_lock_y: Option<Text<String>>,
    spin_lock_a: Option<Text<String>>,
    spin_lock_b: Option<Text<String>>,
    spin_lock_l: Option<Text<String>>,
    spin_lock_r: Option<Text<String>>,
    spin_lock_select: Option<Text<String>>,
    spin_lock_start: Option<Text<String>>,
    quick_reload_left: Option<Text<String>>,
    quick_reload_right: Option<Text<String>>,
    quick_reload_up: Option<Text<String>>,
    quick_reload_down: Option<Text<String>>,
    quick_reload_x: Option<Text<String>>,
    quick_reload_y: Option<Text<String>>,
    quick_reload_a: Option<Text<String>>,
    quick_reload_b: Option<Text<String>>,
    quick_reload_l: Option<Text<String>>,
    quick_reload_r: Option<Text<String>>,
    quick_reload_select: Option<Text<String>>,
    quick_reload_start: Option<Text<String>>,
    moonwalk: Text<bool>,
}

fn upgrade_randomization(randomization: &mut Randomization) {
    if randomization.map.room_mask.is_empty() {
        // For older seeds, room_mask is not specified, and it means all rooms
        // are present:
        randomization.map.room_mask = vec![true; randomization.map.rooms.len()];
    }
}

#[post("/{name}/customize")]
async fn customize_seed(
    req: MultipartForm<CustomizeRequest>,
    info: web::Path<(String,)>,
    app_data: web::Data<AppData>,
) -> impl Responder {
    let seed_name = &info.0;
    let orig_rom = Rom::new(req.rom.data.to_vec());
    let mut rom = orig_rom.clone();

    let seed_data_str: String = String::from_utf8(
        app_data
            .seed_repository
            .get_file(seed_name, "seed_data.json")
            .await
            .unwrap(),
    )
    .unwrap();
    let seed_data = json::parse(&seed_data_str).unwrap();

    let settings_bytes = app_data
        .seed_repository
        .get_file(seed_name, "public/settings.json")
        .await
        .unwrap_or(vec![]);
    let settings: Option<RandomizerSettings> = if settings_bytes.is_empty() {
        None
    } else {
        match try_upgrade_settings(
            String::from_utf8(settings_bytes).unwrap(),
            &app_data.preset_data,
            false,
        ) {
            Ok(s) => Some(s.1),
            Err(e) => {
                return HttpResponse::InternalServerError().body(e.to_string());
            }
        }
    };

    let randomization_bytes = app_data
        .seed_repository
        .get_file(seed_name, "randomization.json")
        .await
        .unwrap_or(vec![]);
    let randomization: Option<Randomization> = if randomization_bytes.is_empty() {
        None
    } else {
        Some(serde_json::from_slice(&randomization_bytes).unwrap())
    };

    let ultra_low_qol = seed_data["ultra_low_qol"].as_bool().unwrap_or(false);

    let rom_digest = crypto_hash::hex_digest(crypto_hash::Algorithm::SHA256, &rom.data);
    info!("Rom digest: {rom_digest}");
    if rom_digest != "12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72" {
        return HttpResponse::BadRequest().body(InvalidRomTemplate {}.render().unwrap());
    }

    let customize_settings = CustomizeSettings {
        samus_sprite: if ultra_low_qol
            && req.samus_sprite.0 == "samus_vanilla"
            && req.vanilla_screw_attack_animation.0
        {
            None
        } else {
            Some(req.samus_sprite.0.clone())
        },
        etank_color: Some((
            u8::from_str_radix(&req.etank_color.0[0..2], 16).unwrap() / 8,
            u8::from_str_radix(&req.etank_color.0[2..4], 16).unwrap() / 8,
            u8::from_str_radix(&req.etank_color.0[4..6], 16).unwrap() / 8,
        )),
        item_dot_change: match req.item_dot_change.0.as_str() {
            "Stay" => maprando::customize::ItemDotChange::Stay,
            "Fade" => maprando::customize::ItemDotChange::Fade,
            "Disappear" => maprando::customize::ItemDotChange::Disappear,
            _ => panic!("Unexpected item_dot_change"),
        },
        transition_letters: req.transition_letters.0,
        reserve_hud_style: req.reserve_hud_style.0,
        vanilla_screw_attack_animation: req.vanilla_screw_attack_animation.0,
        room_names: req.room_names.0,
        palette_theme: if req.room_palettes.0 == "area-themed" {
            PaletteTheme::AreaThemed
        } else {
            PaletteTheme::Vanilla
        },
        tile_theme: if req.tile_theme.0 == "none" {
            TileTheme::Vanilla
        } else if req.tile_theme.0 == "scrambled" {
            TileTheme::Scrambled
        } else if req.tile_theme.0 == "area_themed" {
            TileTheme::AreaThemed
        } else {
            TileTheme::Constant(req.tile_theme.0.to_string())
        },
        door_theme: match req.door_theme.0.as_str() {
            "vanilla" => DoorTheme::Vanilla,
            "alternate" => DoorTheme::Alternate,
            _ => panic!(
                "Unexpected door_theme option: {}",
                req.door_theme.0.as_str()
            ),
        },
        music: match req.music.0.as_str() {
            "area" => MusicSettings::AreaThemed,
            "disabled" => MusicSettings::Disabled,
            _ => panic!("Unexpected music option: {}", req.music.0.as_str()),
        },
        disable_beeping: req.disable_beeping.0,
        shaking: match req.shaking.0.as_str() {
            "Vanilla" => ShakingSetting::Vanilla,
            "Reduced" => ShakingSetting::Reduced,
            "Disabled" => ShakingSetting::Disabled,
            _ => panic!("Unexpected shaking option: {}", req.shaking.0.as_str()),
        },
        flashing: match req.flashing.0.as_str() {
            "Vanilla" => FlashingSetting::Vanilla,
            "Reduced" => FlashingSetting::Reduced,
            _ => panic!("Unexpected flashing option: {}", req.flashing.0.as_str()),
        },
        boss_icons: req.boss_icons.0,
        miniboss_icons: req.miniboss_icons.0,
        save_icons: req.save_icons.0,
        statues_hallway_tiling: match req.statues_hallway_tiling.0.as_str() {
            "Disabled" => StatuesHallwayTiling::Disabled,
            "Default" => StatuesHallwayTiling::Default,
            "Enabled" => StatuesHallwayTiling::Enabled,
            _ => panic!(
                "Unexpected statues_hallway_tiling option: {}",
                req.statues_hallway_tiling.0.as_str()
            ),
        },
        statues_hallway_audio: match req.statues_hallway_audio.0.as_str() {
            "Disabled" => StatuesHallwayAudio::Disabled,
            "Enabled" => StatuesHallwayAudio::Enabled,
            "Louder" => StatuesHallwayAudio::Louder,
            _ => panic!(
                "Unexpected statues_hallway_audio option: {}",
                req.statues_hallway_audio.0.as_str()
            ),
        },
        controller_config: ControllerConfig {
            shot: parse_controller_button(&req.control_shot.0).unwrap(),
            jump: parse_controller_button(&req.control_jump.0).unwrap(),
            dash: parse_controller_button(&req.control_dash.0).unwrap(),
            item_select: parse_controller_button(&req.control_item_select.0).unwrap(),
            item_cancel: parse_controller_button(&req.control_item_cancel.0).unwrap(),
            angle_up: parse_controller_button(&req.control_angle_up.0).unwrap(),
            angle_down: parse_controller_button(&req.control_angle_down.0).unwrap(),
            spin_lock_buttons: get_spin_lock_buttons(&req),
            quick_reload_buttons: get_quick_reload_buttons(&req),
            moonwalk: req.moonwalk.0,
        },
    };

    if settings.is_some()
        && let Some(mut randomization) = randomization
    {
        info!("Patching ROM");
        upgrade_randomization(&mut randomization);
        match make_rom(
            &rom,
            settings.as_ref().unwrap(),
            &customize_settings,
            &randomization,
            &app_data.game_data,
            &app_data.samus_sprite_categories,
            &app_data.mosaic_themes,
        ) {
            Ok(r) => {
                rom = r;
            }
            Err(err) => {
                return HttpResponse::InternalServerError()
                    .body(format!("Error patching ROM: {err:?}"));
            }
        }
    } else {
        return HttpResponse::InternalServerError()
            .body("Seed incompatible with current customizer");
    }
    HttpResponse::Ok()
        .content_type("application/octet-stream")
        .insert_header(ContentDisposition {
            disposition: DispositionType::Attachment,
            parameters: vec![DispositionParam::Filename(
                "map-rando-".to_string() + seed_name + ".sfc",
            )],
        })
        .body(rom.data)
}

fn get_spin_lock_buttons(req: &CustomizeRequest) -> Vec<ControllerButton> {
    let mut spin_lock_buttons = vec![];
    let setting_button_mapping = vec![
        (&req.spin_lock_left, ControllerButton::Left),
        (&req.spin_lock_right, ControllerButton::Right),
        (&req.spin_lock_up, ControllerButton::Up),
        (&req.spin_lock_down, ControllerButton::Down),
        (&req.spin_lock_a, ControllerButton::A),
        (&req.spin_lock_b, ControllerButton::B),
        (&req.spin_lock_x, ControllerButton::X),
        (&req.spin_lock_y, ControllerButton::Y),
        (&req.spin_lock_l, ControllerButton::L),
        (&req.spin_lock_r, ControllerButton::R),
        (&req.spin_lock_select, ControllerButton::Select),
        (&req.spin_lock_start, ControllerButton::Start),
    ];

    for (setting, button) in setting_button_mapping {
        if let Some(x) = setting
            && x.0 == "on"
        {
            spin_lock_buttons.push(button);
        }
    }
    spin_lock_buttons
}

fn get_quick_reload_buttons(req: &CustomizeRequest) -> Vec<ControllerButton> {
    let mut quick_reload_buttons = vec![];
    let setting_button_mapping = vec![
        (&req.quick_reload_left, ControllerButton::Left),
        (&req.quick_reload_right, ControllerButton::Right),
        (&req.quick_reload_up, ControllerButton::Up),
        (&req.quick_reload_down, ControllerButton::Down),
        (&req.quick_reload_a, ControllerButton::A),
        (&req.quick_reload_b, ControllerButton::B),
        (&req.quick_reload_x, ControllerButton::X),
        (&req.quick_reload_y, ControllerButton::Y),
        (&req.quick_reload_l, ControllerButton::L),
        (&req.quick_reload_r, ControllerButton::R),
        (&req.quick_reload_select, ControllerButton::Select),
        (&req.quick_reload_start, ControllerButton::Start),
    ];

    for (setting, button) in setting_button_mapping {
        if let Some(x) = setting
            && x.0 == "on"
        {
            quick_reload_buttons.push(button);
        }
    }
    quick_reload_buttons
}

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

pub fn scope() -> actix_web::Scope {
    actix_web::web::scope("/seed")
        .service(view_seed)
        .service(view_seed_redirect)
        .service(get_seed_file)
        .service(customize_seed)
        .service(unlock_seed)
}
