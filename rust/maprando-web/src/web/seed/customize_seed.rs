use crate::web::{upgrade::try_upgrade_settings, AppData};
use actix_easy_multipart::{bytes::Bytes, text::Text, MultipartForm};
use actix_web::{
    http::header::{ContentDisposition, DispositionParam, DispositionType},
    post, web, HttpResponse, Responder,
};
use askama::Template;
use log::info;
use maprando::{
    customize::{
        customize_rom, parse_controller_button, ControllerButton, ControllerConfig,
        CustomizeSettings, DoorTheme, FlashingSetting, MusicSettings, Overrides, PaletteTheme,
        ShakingSetting, TileTheme,
    },
    patch::{make_rom, Rom},
    randomize::Randomization,
    settings::{DoorLocksSize, ItemDotChange, RandomizerSettings},
};
use maprando_game::Map;

#[derive(Template)]
#[template(path = "errors/invalid_rom.html")]
struct InvalidRomTemplate {}

#[derive(MultipartForm)]
struct CustomizeRequest {
    rom: Bytes,
    samus_sprite: Text<String>,
    etank_color: Text<String>,
    reserve_hud_style: Text<bool>,
    room_palettes: Text<String>,
    tile_theme: Text<String>,
    door_theme: Text<String>,
    music: Text<String>,
    disable_beeping: Text<bool>,
    shaking: Text<String>,
    flashing: Text<String>,
    vanilla_screw_attack_animation: Text<bool>,
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
    // item_dot_change, transition_letters, and door_locks_size are prefixed with "customize_" to
    // distinguish them from the same options in the RandomizerSettings (settings from the generate
    // page).
    customize_item_dot_change: Text<String>,
    customize_transition_letters: Text<bool>,
    customize_door_locks_size: Text<String>,
    override_mark_map_stations: Text<String>,
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

    let map_data_bytes = app_data
        .seed_repository
        .get_file(seed_name, "map.json")
        .await
        .unwrap_or(vec![]);
    let map: Option<Map> = if map_data_bytes.len() == 0 {
        None
    } else {
        Some(serde_json::from_slice(&map_data_bytes).unwrap())
    };

    let settings_bytes = app_data
        .seed_repository
        .get_file(seed_name, "public/settings.json")
        .await
        .unwrap_or(vec![]);
    let settings: Option<RandomizerSettings> = if settings_bytes.len() == 0 {
        None
    } else {
        match try_upgrade_settings(String::from_utf8(settings_bytes).unwrap(), &app_data, false) {
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
    let randomization: Option<Randomization> = if randomization_bytes.len() == 0 {
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

    let overrides = if settings
        .as_ref()
        .is_some_and(|settings| !settings.other_settings.race_mode)
    {
        Overrides {
            mark_map_stations: match req.override_mark_map_stations.0.as_str() {
                "Unchanged" => None,
                "false" => Some(false),
                "true" => Some(true),
                _ => panic!(
                    "Unexpected override mark map stations option: {}",
                    req.override_mark_map_stations.0.as_str()
                ),
            },
        }
    } else {
        Overrides::default()
    };

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
        reserve_hud_style: req.reserve_hud_style.0,
        vanilla_screw_attack_animation: req.vanilla_screw_attack_animation.0,
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
        item_dot_change: match req.customize_item_dot_change.0.as_str() {
            "Fade" => ItemDotChange::Fade,
            "Disappear" => ItemDotChange::Disappear,
            _ => panic!(
                "Unexpected customize item dot change option: {}",
                req.customize_item_dot_change.0.as_str()
            ),
        },
        transition_letters: req.customize_transition_letters.0,
        door_locks_size: match req.customize_door_locks_size.0.as_str() {
            "Small" => DoorLocksSize::Small,
            "Large" => DoorLocksSize::Large,
            _ => panic!(
                "Unexpected customize door locks size option: {}",
                req.customize_door_locks_size.0.as_str()
            ),
        },
        overrides,
    };

    if let Some((mut settings, randomization)) = settings.zip(randomization) {
        settings.apply_overrides(&customize_settings);
        info!("Patching ROM");
        match make_rom(&rom, &settings, &randomization, &app_data.game_data) {
            Ok(r) => {
                rom = r;
            }
            Err(err) => {
                return HttpResponse::InternalServerError()
                    .body(format!("Error patching ROM: {:?}", err))
            }
        }
    } else {
        return HttpResponse::InternalServerError()
            .body(format!("Seed incompatible with current customizer"));
    }

    info!("CustomizeSettings: {:?}", customize_settings);
    match customize_rom(
        &mut rom,
        &orig_rom,
        &map,
        &customize_settings,
        &app_data.game_data,
        &app_data.samus_sprite_categories,
        &app_data.mosaic_themes,
    ) {
        Ok(()) => {}
        Err(err) => {
            return HttpResponse::InternalServerError()
                .body(format!("Error customizing ROM: {:?}", err))
        }
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
        if let Some(x) = setting {
            if x.0 == "on" {
                spin_lock_buttons.push(button);
            }
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
        if let Some(x) = setting {
            if x.0 == "on" {
                quick_reload_buttons.push(button);
            }
        }
    }
    quick_reload_buttons
}
