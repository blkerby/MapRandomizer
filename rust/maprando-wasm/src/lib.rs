use maprando_logic::{boss_requirements::*, Inventory, LocalState};
use wasm_bindgen::prelude::*;

extern crate console_error_panic_hook;
use std::panic;

#[wasm_bindgen]
pub fn set_panic_hook() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}

#[wasm_bindgen]
pub fn can_defeat_phantoon(
    inventory: JsValue,
    proficiency: f32,
    can_manage_reserves: bool,
) -> Result<bool, JsValue> {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory)?;

    Ok(apply_phantoon_requirement(
        &inventory,
        LocalState::new(),
        proficiency,
        can_manage_reserves,
    )
    .is_some())
}

#[wasm_bindgen]
pub fn can_defeat_draygon(
    inventory: JsValue,
    proficiency: f32,
    can_manage_reserves: bool,
    can_be_very_patient: bool,
) -> Result<bool, JsValue> {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory)?;

    Ok(apply_draygon_requirement(
        &inventory,
        LocalState::new(),
        proficiency,
        can_manage_reserves,
        can_be_very_patient,
    )
    .is_some())
}

#[wasm_bindgen]
pub fn can_defeat_ridley(
    inventory: JsValue,
    proficiency: f32,
    can_manage_reserves: bool,
    can_be_very_patient: bool,
) -> Result<bool, JsValue> {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory)?;

    Ok(apply_ridley_requirement(
        &inventory,
        LocalState::new(),
        proficiency,
        can_manage_reserves,
        can_be_very_patient,
    )
    .is_some())
}

#[wasm_bindgen]
pub fn can_defeat_botwoon(
    inventory: JsValue,
    proficiency: f32,
    second_phase: bool,
    can_manage_reserves: bool,
) -> Result<bool, JsValue> {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory)?;

    Ok(apply_botwoon_requirement(
        &inventory,
        LocalState::new(),
        proficiency,
        second_phase,
        can_manage_reserves,
    )
    .is_some())
}

#[wasm_bindgen]
pub fn can_defeat_mother_brain_2(
    inventory: JsValue,
    proficiency: f32,
    supers_double: bool,
    can_manage_reserves: bool,
    can_be_very_patient: bool,
    r_mode: bool,
) -> Result<bool, JsValue> {
    let inventory: Inventory = serde_wasm_bindgen::from_value(inventory)?;

    Ok(apply_mother_brain_2_requirement(
        &inventory,
        LocalState::new(),
        proficiency,
        supers_double,
        can_manage_reserves,
        can_be_very_patient,
        r_mode,
    )
    .is_some())
}
