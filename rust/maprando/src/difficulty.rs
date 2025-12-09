use hashbrown::HashMap;
use maprando_game::{
    DoorOrientation, ExitCondition, GameData, Link, LinkLength, MainEntranceCondition, NodeId,
    Requirement, RoomId, SparkPosition, TECH_ID_CAN_BOMB_HORIZONTALLY,
    TECH_ID_CAN_CARRY_FLASH_SUIT, TECH_ID_CAN_ENEMY_STUCK_MOONFALL, TECH_ID_CAN_ENTER_G_MODE,
    TECH_ID_CAN_ENTER_G_MODE_IMMOBILE, TECH_ID_CAN_ENTER_R_MODE, TECH_ID_CAN_EXTENDED_MOONDANCE,
    TECH_ID_CAN_GRAPPLE_JUMP, TECH_ID_CAN_GRAPPLE_TELEPORT, TECH_ID_CAN_HEATED_G_MODE,
    TECH_ID_CAN_HORIZONTAL_SHINESPARK, TECH_ID_CAN_MIDAIR_SHINESPARK, TECH_ID_CAN_MOCKBALL,
    TECH_ID_CAN_MOONDANCE, TECH_ID_CAN_PRECISE_GRAPPLE, TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK,
    TECH_ID_CAN_SAMUS_EATER_TELEPORT, TECH_ID_CAN_SHINECHARGE_MOVEMENT,
    TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP, TECH_ID_CAN_SPEEDBALL,
    TECH_ID_CAN_SPRING_BALL_BOUNCE, TECH_ID_CAN_STATIONARY_SPIN_JUMP,
    TECH_ID_CAN_STUTTER_WATER_SHINECHARGE, TECH_ID_CAN_SUPER_SINK, TECH_ID_CAN_TEMPORARY_BLUE,
    TECH_ID_CAN_WALLJUMP, VertexAction,
};
use maprando_logic::{GlobalState, Inventory, LocalState};

use crate::{
    preset::PresetData,
    traverse::{LockedDoorData, apply_requirement, simple_cost_config},
};

pub fn get_full_global(game_data: &GameData) -> GlobalState {
    let items = vec![true; game_data.item_isv.keys.len()];
    let tech = vec![true; game_data.tech_isv.keys.len()];
    let weapon_mask = game_data.get_weapon_mask(&items, &tech);
    let inventory = Inventory {
        items,
        max_energy: 1499,
        max_reserves: 400,
        max_missiles: 230,
        max_supers: 50,
        max_power_bombs: 50,
        collectible_missile_packs: 0,
        collectible_super_packs: 0,
        collectible_power_bomb_packs: 0,
        collectible_reserve_tanks: 0,
    };
    GlobalState {
        pool_inventory: inventory.clone(),
        inventory,
        flags: vec![true; game_data.flag_isv.keys.len()],
        doors_unlocked: vec![],
        weapon_mask,
    }
}

pub fn get_link_difficulty(
    link: &Link,
    game_data: &GameData,
    preset_data: &PresetData,
    global: &GlobalState,
) -> u8 {
    let extra_req = get_cross_room_reqs(link, game_data);
    let main_req = strip_cross_room_reqs(link.requirement.clone());
    let combined_req = Requirement::make_and(vec![extra_req, main_req]);

    let locked_door_data = LockedDoorData {
        locked_doors: vec![],
        locked_door_node_map: HashMap::new(),
        locked_door_vertex_ids: vec![],
    };
    let door_map: HashMap<(RoomId, NodeId), (RoomId, NodeId)> = HashMap::new();
    let cost_config = simple_cost_config();

    for (i, difficulty) in preset_data.difficulty_tiers.iter().rev().enumerate() {
        let difficulty_idx = preset_data.difficulty_levels.index_by_key[&difficulty.name];

        let flash_suit_obtainable =
            difficulty.tech[game_data.tech_isv.index_by_key[&TECH_ID_CAN_CARRY_FLASH_SUIT]];

        let local = LocalState {
            shinecharge_frames_remaining: 180 - difficulty.shinecharge_leniency_frames,
            flash_suit: if flash_suit_obtainable { 1 } else { 0 },
            ..LocalState::full(false)
        };

        let new_local = apply_requirement(
            &combined_req,
            global,
            local,
            false,
            &preset_data.logic_page_presets[difficulty_idx],
            difficulty,
            game_data,
            &door_map,
            &locked_door_data,
            &[],
            &cost_config,
        );
        if new_local.is_some() {
            return i as u8;
        }
    }
    preset_data.difficulty_tiers.len() as u8
}

pub fn get_link_difficulty_length(
    link: &Link,
    game_data: &GameData,
    preset_data: &PresetData,
    global: &GlobalState,
) -> (u8, LinkLength) {
    let tier = get_link_difficulty(link, game_data, preset_data, global);
    let mut difficulty = 1 << tier;
    if !link.strat_name.starts_with("Base ") && link.strat_name != "Base" {
        difficulty += 1;
    }
    (tier, difficulty)
}

fn strip_cross_room_reqs(req: Requirement) -> Requirement {
    match req {
        Requirement::And(subreqs) => {
            Requirement::make_and(subreqs.into_iter().map(strip_cross_room_reqs).collect())
        }
        Requirement::Or(subreqs) => {
            Requirement::make_or(subreqs.into_iter().map(strip_cross_room_reqs).collect())
        }
        Requirement::DoorUnlocked { .. } => Requirement::Free,
        Requirement::NotFlag(_) => Requirement::Free,
        _ => req,
    }
}

fn get_cross_room_reqs(link: &Link, game_data: &GameData) -> Requirement {
    let mut reqs: Vec<Requirement> = vec![];
    let from_vertex_key = &game_data.vertex_isv.keys[link.from_vertex_id];
    let to_vertex_key = &game_data.vertex_isv.keys[link.to_vertex_id];

    // TODO: handle gModeRegainMobility in some cleaner way:
    for (l, _) in game_data
        .node_gmode_regain_mobility
        .get(&(from_vertex_key.room_id, from_vertex_key.node_id))
        .unwrap_or(&vec![])
    {
        if l.strat_id.is_some() && l.strat_id == link.strat_id {
            reqs.push(Requirement::Tech(
                game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE_IMMOBILE],
            ));
            if game_data
                .get_room_heated(
                    &game_data.room_json_map[&from_vertex_key.room_id],
                    from_vertex_key.node_id,
                )
                .unwrap()
            {
                reqs.push(Requirement::Tech(
                    game_data.tech_isv.index_by_key[&TECH_ID_CAN_HEATED_G_MODE],
                ));
            }
        }
    }

    for action in &from_vertex_key.actions {
        if let VertexAction::Enter(entrance_condition) = action {
            match &entrance_condition.main {
                MainEntranceCondition::ComeInNormally { .. } => {}
                MainEntranceCondition::ComeInRunning { .. } => {}
                MainEntranceCondition::ComeInJumping { .. } => {}
                MainEntranceCondition::ComeInSpaceJumping { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                    ));
                }
                MainEntranceCondition::ComeInBlueSpaceJumping { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                    ));
                }
                MainEntranceCondition::ComeInGettingBlueSpeed { .. } => {}
                MainEntranceCondition::ComeInShinecharging { .. } => {}
                MainEntranceCondition::ComeInShinecharged { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SHINECHARGE_MOVEMENT],
                    ));
                }
                MainEntranceCondition::ComeInShinechargedJumping { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SHINECHARGE_MOVEMENT],
                    ));
                }
                MainEntranceCondition::ComeInWithSpark {
                    position,
                    door_orientation,
                } => {
                    if [DoorOrientation::Left, DoorOrientation::Right].contains(door_orientation) {
                        reqs.push(Requirement::Tech(
                            game_data.tech_isv.index_by_key[&TECH_ID_CAN_HORIZONTAL_SHINESPARK],
                        ));
                        if position == &SparkPosition::Top {
                            reqs.push(Requirement::Tech(
                                game_data.tech_isv.index_by_key[&TECH_ID_CAN_MIDAIR_SHINESPARK],
                            ));
                        }
                    }
                }
                MainEntranceCondition::ComeInWithBombBoost {} => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_BOMB_HORIZONTALLY],
                    ));
                }
                MainEntranceCondition::ComeInStutterShinecharging { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_STUTTER_WATER_SHINECHARGE],
                    ));
                }
                MainEntranceCondition::ComeInStutterGettingBlueSpeed { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_STUTTER_WATER_SHINECHARGE],
                    ));
                }
                MainEntranceCondition::ComeInWithDoorStuckSetup {
                    door_orientation, ..
                } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_STATIONARY_SPIN_JUMP],
                    ));
                    if door_orientation == &DoorOrientation::Right {
                        reqs.push(Requirement::Tech(
                            game_data.tech_isv.index_by_key[&TECH_ID_CAN_RIGHT_SIDE_DOOR_STUCK],
                        ));
                    }
                }
                MainEntranceCondition::ComeInSpeedballing { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPEEDBALL],
                    ));
                }
                MainEntranceCondition::ComeInWithTemporaryBlue { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_TEMPORARY_BLUE],
                    ));
                }
                MainEntranceCondition::ComeInWithMockball { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOCKBALL],
                    ));
                }
                MainEntranceCondition::ComeInWithSpringBallBounce { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPRING_BALL_BOUNCE],
                    ));
                }
                MainEntranceCondition::ComeInWithBlueSpringBallBounce { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPRING_BALL_BOUNCE],
                    ));
                }
                MainEntranceCondition::ComeInSpinning { .. } => {}
                MainEntranceCondition::ComeInBlueSpinning { .. } => {}
                MainEntranceCondition::ComeInWithStoredFallSpeed {
                    fall_speed_in_tiles,
                } => {
                    reqs.push(Requirement::Or(vec![
                        Requirement::Tech(game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOONDANCE]),
                        Requirement::Tech(
                            game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENEMY_STUCK_MOONFALL],
                        ),
                    ]));
                    if fall_speed_in_tiles == &2 {
                        reqs.push(Requirement::Tech(
                            game_data.tech_isv.index_by_key[&TECH_ID_CAN_EXTENDED_MOONDANCE],
                        ));
                    }
                }
                MainEntranceCondition::ComeInWithRMode { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_R_MODE],
                    ));
                }
                MainEntranceCondition::ComeInWithGMode { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE],
                    ));
                }
                MainEntranceCondition::ComeInWithWallJumpBelow { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_WALLJUMP],
                    ));
                }
                MainEntranceCondition::ComeInWithSpaceJumpBelow { .. } => {}
                MainEntranceCondition::ComeInWithPlatformBelow { .. } => {}
                MainEntranceCondition::ComeInWithSidePlatform { platforms } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                    ));
                    for p in platforms {
                        reqs.push(p.requirement.clone());
                    }
                }
                MainEntranceCondition::ComeInWithGrappleSwing { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_PRECISE_GRAPPLE],
                    ));
                }
                MainEntranceCondition::ComeInWithGrappleJump { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_JUMP],
                    ));
                }
                MainEntranceCondition::ComeInWithGrappleTeleport { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_TELEPORT],
                    ));
                }
                MainEntranceCondition::ComeInWithSamusEaterTeleport { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SAMUS_EATER_TELEPORT],
                    ));
                }
                MainEntranceCondition::ComeInWithSuperSink { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SUPER_SINK],
                    ));
                }
            }
        }
    }
    for action in &to_vertex_key.actions {
        if let VertexAction::Exit(exit_condition) = action {
            match exit_condition {
                ExitCondition::LeaveNormally { .. } => {}
                ExitCondition::LeaveWithRunway { .. } => {}
                ExitCondition::LeaveShinecharged { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SHINECHARGE_MOVEMENT],
                    ));
                }
                ExitCondition::LeaveWithTemporaryBlue { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_TEMPORARY_BLUE],
                    ));
                }
                ExitCondition::LeaveWithSpark {
                    position,
                    door_orientation,
                } => {
                    if [DoorOrientation::Left, DoorOrientation::Right].contains(door_orientation) {
                        reqs.push(Requirement::Tech(
                            game_data.tech_isv.index_by_key[&TECH_ID_CAN_HORIZONTAL_SHINESPARK],
                        ));
                        if position == &SparkPosition::Top {
                            reqs.push(Requirement::Tech(
                                game_data.tech_isv.index_by_key[&TECH_ID_CAN_MIDAIR_SHINESPARK],
                            ));
                        }
                    }
                }
                ExitCondition::LeaveSpinning { .. } => {}
                ExitCondition::LeaveWithMockball { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_MOCKBALL],
                    ));
                }
                ExitCondition::LeaveWithSpringBallBounce { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SPRING_BALL_BOUNCE],
                    ));
                }
                ExitCondition::LeaveSpaceJumping { .. } => {}
                ExitCondition::LeaveWithStoredFallSpeed { .. } => {}
                ExitCondition::LeaveWithGModeSetup { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE],
                    ));
                }
                ExitCondition::LeaveWithGMode { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_ENTER_G_MODE],
                    ));
                }
                ExitCondition::LeaveWithDoorFrameBelow { .. } => {}
                ExitCondition::LeaveWithPlatformBelow { .. } => {}
                ExitCondition::LeaveWithSidePlatform { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SIDE_PLATFORM_CROSS_ROOM_JUMP],
                    ));
                }
                ExitCondition::LeaveWithGrappleSwing { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_PRECISE_GRAPPLE],
                    ));
                }
                ExitCondition::LeaveWithGrappleJump { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_JUMP],
                    ));
                }
                ExitCondition::LeaveWithGrappleTeleport { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_GRAPPLE_TELEPORT],
                    ));
                }
                ExitCondition::LeaveWithSamusEaterTeleport { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SAMUS_EATER_TELEPORT],
                    ));
                }
                ExitCondition::LeaveWithSuperSink { .. } => {
                    reqs.push(Requirement::Tech(
                        game_data.tech_isv.index_by_key[&TECH_ID_CAN_SUPER_SINK],
                    ));
                }
            }
        }
    }
    Requirement::make_and(reqs)
}
