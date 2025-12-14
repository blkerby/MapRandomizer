use maprando_game::RidleyStuck;
use maprando_logic::{Inventory, LocalState, boss_requirements::*};
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
pub fn can_defeat_phantoon(inventory: JsValue, local: JsValue, proficiency: f32) -> JsValue {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory).unwrap();
    let mut local =
        serde_wasm_bindgen::from_value(local).unwrap_or_else(|_| LocalState::full(false));

    if apply_phantoon_requirement(&inventory, &mut local, proficiency, false) {
        serde_wasm_bindgen::to_value(&local).unwrap()
    } else {
        JsValue::null()
    }
}

#[wasm_bindgen]
pub fn can_defeat_draygon(
    inventory: JsValue,
    local: JsValue,
    proficiency: f32,
    can_be_patient: bool,
    can_be_very_patient: bool,
    can_be_extremely_patient: bool,
) -> JsValue {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory).unwrap();
    let mut local =
        serde_wasm_bindgen::from_value(local).unwrap_or_else(|_| LocalState::full(false));

    if apply_draygon_requirement(
        &inventory,
        &mut local,
        proficiency,
        can_be_patient,
        can_be_very_patient,
        can_be_extremely_patient,
        false,
    ) {
        serde_wasm_bindgen::to_value(&local).unwrap()
    } else {
        JsValue::null()
    }
}

#[wasm_bindgen]
pub fn can_defeat_ridley(
    inventory: JsValue,
    local: JsValue,
    proficiency: f32,
    can_be_patient: bool,
    can_be_very_patient: bool,
    can_be_extremely_patient: bool,
) -> JsValue {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory).unwrap();
    let mut local =
        serde_wasm_bindgen::from_value(local).unwrap_or_else(|_| LocalState::full(false));

    if apply_ridley_requirement(
        &inventory,
        &mut local,
        proficiency,
        can_be_patient,
        can_be_very_patient,
        can_be_extremely_patient,
        true,
        false,
        RidleyStuck::None,
        false,
    ) {
        serde_wasm_bindgen::to_value(&local).unwrap()
    } else {
        JsValue::null()
    }
}

#[wasm_bindgen]
pub fn can_defeat_botwoon(
    inventory: JsValue,
    local: JsValue,
    proficiency: f32,
    second_phase: bool,
) -> JsValue {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory).unwrap();
    let mut local =
        serde_wasm_bindgen::from_value(local).unwrap_or_else(|_| LocalState::full(false));

    if apply_botwoon_requirement(&inventory, &mut local, proficiency, second_phase, false) {
        serde_wasm_bindgen::to_value(&local).unwrap()
    } else {
        JsValue::null()
    }
}

#[wasm_bindgen]
pub fn can_defeat_mother_brain_2(
    inventory: JsValue,
    local: JsValue,
    proficiency: f32,
    supers_double: bool,
    can_be_very_patient: bool,
    r_mode: bool,
) -> JsValue {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory).unwrap();
    let mut local =
        serde_wasm_bindgen::from_value(local).unwrap_or_else(|_| LocalState::full(false));

    if apply_mother_brain_2_requirement(
        &inventory,
        &mut local,
        proficiency,
        supers_double,
        can_be_very_patient,
        r_mode,
        false,
    ) {
        serde_wasm_bindgen::to_value(&local).unwrap()
    } else {
        JsValue::null()
    }
}
