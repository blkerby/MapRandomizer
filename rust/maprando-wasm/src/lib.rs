use maprando_game::RidleyStuck;
use maprando_logic::{boss_requirements::*, Inventory};
use wasm_bindgen::prelude::*;

extern crate console_error_panic_hook;
use std::panic;

#[wasm_bindgen(start)]
pub fn startup() -> Result<(), JsValue> {
    wasm_logger::init(wasm_logger::Config::default());
    Ok(())
}

#[wasm_bindgen]
pub fn set_panic_hook() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}

#[wasm_bindgen]
pub fn can_defeat_phantoon(
    inventory: JsValue,
    local: JsValue,
    proficiency: f32,
    can_manage_reserves: bool,
) -> JsValue {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory).unwrap();
    let local = serde_wasm_bindgen::from_value(local).unwrap_or_default();

    match apply_phantoon_requirement(&inventory, local, proficiency, can_manage_reserves) {
        Some(local) => serde_wasm_bindgen::to_value(&local).unwrap(),
        None => JsValue::null(),
    }
}

#[wasm_bindgen]
pub fn can_defeat_draygon(
    inventory: JsValue,
    local: JsValue,
    proficiency: f32,
    can_manage_reserves: bool,
    can_be_very_patient: bool,
) -> JsValue {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory).unwrap();
    let local = serde_wasm_bindgen::from_value(local).unwrap_or_default();

    match apply_draygon_requirement(
        &inventory,
        local,
        proficiency,
        can_manage_reserves,
        can_be_very_patient,
    ) {
        Some(local) => serde_wasm_bindgen::to_value(&local).unwrap(),
        None => JsValue::null(),
    }
}

#[wasm_bindgen]
pub fn can_defeat_ridley(
    inventory: JsValue,
    local: JsValue,
    proficiency: f32,
    can_manage_reserves: bool,
    can_be_patient: bool,
    can_be_very_patient: bool,
    can_be_extremely_patient: bool,
) -> JsValue {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory).unwrap();
    let local = serde_wasm_bindgen::from_value(local).unwrap_or_default();

    match apply_ridley_requirement(
        &inventory,
        local,
        proficiency,
        can_manage_reserves,
        can_be_patient,
        can_be_very_patient,
        can_be_extremely_patient,
        true,
        false,
        RidleyStuck::None,
    ) {
        Some(local) => serde_wasm_bindgen::to_value(&local).unwrap(),
        None => JsValue::null(),
    }
}

#[wasm_bindgen]
pub fn can_defeat_botwoon(
    inventory: JsValue,
    local: JsValue,
    proficiency: f32,
    second_phase: bool,
    can_manage_reserves: bool,
) -> JsValue {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory).unwrap();
    let local = serde_wasm_bindgen::from_value(local).unwrap_or_default();

    match apply_botwoon_requirement(
        &inventory,
        local,
        proficiency,
        second_phase,
        can_manage_reserves,
    ) {
        Some(local) => serde_wasm_bindgen::to_value(&local).unwrap(),
        None => JsValue::null(),
    }
}

#[wasm_bindgen]
pub fn can_defeat_mother_brain_2(
    inventory: JsValue,
    local: JsValue,
    proficiency: f32,
    supers_double: bool,
    can_manage_reserves: bool,
    can_be_very_patient: bool,
    r_mode: bool,
) -> JsValue {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory).unwrap();
    let local = serde_wasm_bindgen::from_value(local).unwrap_or_default();

    match apply_mother_brain_2_requirement(
        &inventory,
        local,
        proficiency,
        supers_double,
        can_manage_reserves,
        can_be_very_patient,
        r_mode,
    ) {
        Some(local) => serde_wasm_bindgen::to_value(&local).unwrap(),
        None => JsValue::null(),
    }
}
