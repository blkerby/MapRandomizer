use std::{
    collections::BTreeSet,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    process::Command,
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use hashbrown::HashMap;
use maprando::{
    preset::PresetData,
    randomize::DifficultyConfig,
    settings::WallJump,
    traverse::{LockedDoorData, Traverser, apply_requirement, simple_cost_config},
};
use maprando_game::{
    ExitCondition, GameData, LinksDataGroup, MainEntranceCondition, NodeId, Requirement, RoomId,
    VertexId, VertexKey,
};
use maprando_logic::{GlobalState, Inventory, LocalState};
use serde::Serialize;

#[derive(Parser)]
#[command(about = "Extract approximate room-local, empty-inventory reachability graphs")]
struct Args {
    #[arg(long)]
    output: std::path::PathBuf,
}

#[derive(Serialize)]
struct Node {
    room_id: RoomId,
    node_id: NodeId,
    name: String,
    door: bool,
    item: bool,
    start: bool,
}

struct Endpoint {
    node: Node,
    sources: Vec<(VertexId, Requirement)>,
    targets: Vec<(VertexId, Requirement)>,
}

#[derive(Serialize)]
struct Graph {
    difficulty: String,
    wall_jump: bool,
    edges: BTreeSet<(usize, usize)>,
}

#[derive(Serialize)]
struct Source {
    revision: String,
    tracked_changes: bool,
}

fn source(path: &Path) -> Result<Source> {
    let revision = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(path)
        .output()?;
    ensure!(
        revision.status.success(),
        "Cannot read revision at {path:?}"
    );
    let status = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .current_dir(path)
        .output()?;
    ensure!(status.status.success(), "Cannot read status at {path:?}");
    Ok(Source {
        revision: String::from_utf8(revision.stdout)?.trim().to_string(),
        tracked_changes: !status.stdout.is_empty(),
    })
}

fn endpoints(game: &GameData) -> Vec<Endpoint> {
    let mut doors: BTreeSet<_> = game.node_entrance_conditions.keys().copied().collect();
    doors.extend(game.node_exit_conditions.keys().copied());
    let items: HashMap<_, _> = game
        .item_locations
        .iter()
        .copied()
        .zip(&game.item_vertex_ids)
        .collect();
    let mut nodes = doors.clone();
    nodes.extend(items.keys().copied());
    nodes.extend(game.start_location_id_map.keys().copied());
    nodes.insert((8, 5)); // Ship start
    let mut out = Vec::new();
    for (room_id, node_id) in nodes {
        let mut sources: Vec<_> = game
            .node_entrance_conditions
            .get(&(room_id, node_id))
            .into_iter()
            .flatten()
            .filter(|(_, entrance)| {
                matches!(entrance.main, MainEntranceCondition::ComeInNormally {})
            })
            .map(|(vertex, _)| (*vertex, Requirement::Free))
            .collect();
        let mut targets: Vec<_> = game
            .node_exit_conditions
            .get(&(room_id, node_id))
            .into_iter()
            .flatten()
            .filter(|exit| matches!(exit.exit_condition, ExitCondition::LeaveNormally {}))
            .map(|exit| (exit.vertex_id, exit.exit_req.clone()))
            .collect();
        if let Some(vertices) = items.get(&(room_id, node_id)) {
            sources.extend(vertices.iter().map(|&vertex| (vertex, Requirement::Free)));
            targets.extend(vertices.iter().map(|&vertex| (vertex, Requirement::Free)));
        }
        let start_requirement = if (room_id, node_id) == (8, 5) {
            Some(Requirement::Free)
        } else {
            game.start_location_id_map
                .get(&(room_id, node_id))
                .map(|&idx| game.start_locations[idx].requires_parsed.clone().unwrap())
        };
        let start = start_requirement.is_some();
        if let Some(requirement) = start_requirement {
            let vertex = game.vertex_isv.index_by_key[&VertexKey {
                room_id,
                node_id,
                obstacle_mask: 0,
                actions: vec![],
            }];
            sources.push((vertex, requirement));
            targets.push((vertex, Requirement::Free));
        }
        out.push(Endpoint {
            node: Node {
                room_id,
                node_id,
                name: game.node_json_map[&(room_id, node_id)]["name"]
                    .as_str()
                    .unwrap()
                    .to_string(),
                door: doors.contains(&(room_id, node_id)),
                item: items.contains_key(&(room_id, node_id)),
                start,
            },
            sources,
            targets,
        });
    }
    out
}

fn extract(
    game: &GameData,
    preset: &PresetData,
    difficulty: &DifficultyConfig,
    endpoints: &[Endpoint],
    wall_jump: bool,
) -> Graph {
    let mut settings = preset.default_preset.clone();
    settings.other_settings.wall_jump = if wall_jump {
        WallJump::Vanilla
    } else {
        WallJump::Collectible
    };
    let inventory = Inventory {
        items: vec![false; game.item_isv.keys.len()],
        max_energy: 99,
        max_reserves: 0,
        max_missiles: 0,
        max_supers: 0,
        max_power_bombs: 0,
        collectible_missile_packs: 0,
        collectible_super_packs: 0,
        collectible_power_bomb_packs: 0,
        collectible_reserve_tanks: 0,
    };
    let global = GlobalState {
        weapon_mask: game.get_weapon_mask(&inventory.items, &difficulty.tech),
        pool_inventory: inventory.clone(),
        inventory,
        flags: vec![false; game.flag_isv.keys.len()],
        doors_unlocked: vec![],
    };
    let door_map = HashMap::new();
    let locked_doors = LockedDoorData {
        locked_doors: vec![],
        locked_door_node_map: HashMap::new(),
        locked_door_vertex_ids: vec![],
    };
    let cost = simple_cost_config();
    let num_vertices = game.vertex_isv.keys.len();
    let empty_links = LinksDataGroup::new(vec![], num_vertices, game.links.len());
    let mut traverser = Traverser::new(num_vertices, false, LocalState::full(false), &global);
    let mut edges = BTreeSet::new();
    for (from, endpoint) in endpoints.iter().enumerate() {
        for (vertex, requirement) in &endpoint.sources {
            if let Some(initial_local) = apply_requirement(
                requirement,
                &global,
                LocalState::full(false),
                false,
                &settings,
                difficulty,
                game,
                &door_map,
                &locked_doors,
                &[],
                &cost,
            ) {
                traverser.add_origin(initial_local, &global.inventory, *vertex);
            }
        }
        traverser.traverse(
            &game.base_links_data,
            &empty_links,
            &global,
            &settings,
            difficulty,
            game,
            &door_map,
            &locked_doors,
            &[],
            0,
        );
        for (to, target) in endpoints.iter().enumerate() {
            if from == to || target.node.room_id != endpoint.node.room_id {
                continue;
            }
            if target.targets.iter().any(|(vertex, requirement)| {
                traverser.lsr[*vertex].local.iter().any(|&local| {
                    apply_requirement(
                        requirement,
                        &global,
                        local,
                        false,
                        &settings,
                        difficulty,
                        game,
                        &door_map,
                        &locked_doors,
                        &[],
                        &cost,
                    )
                    .is_some()
                })
            }) {
                edges.insert((from, to));
            }
        }
        traverser.pop_step();
    }
    Graph {
        difficulty: difficulty.name.clone(),
        wall_jump,
        edges,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut game = GameData::load(Path::new("."))?;
    game.make_links_data(&|_, _| (0, 1));
    let preset = PresetData::load(
        Path::new("data/tech_data.json"),
        Path::new("data/notable_data.json"),
        Path::new("data/presets"),
        &game,
    )?;
    let endpoints = endpoints(&game);
    let mut graphs = Vec::new();
    for difficulty in preset.difficulty_tiers.iter().rev() {
        for wall_jump in [false, true] {
            graphs.push(extract(&game, &preset, difficulty, &endpoints, wall_jump));
        }
    }
    let nodes: Vec<_> = endpoints.iter().map(|e| &e.node).collect();
    let rooms: std::collections::BTreeMap<_, _> = game
        .room_json_map
        .iter()
        .map(|(&id, room)| (id, room["name"].as_str().unwrap()))
        .collect();
    let output = serde_json::json!({
        "format": "initial-item-graph-v1",
        "sources": { "map_randomizer": source(Path::new("."))?, "sm_json_data": source(Path::new("../sm-json-data"))? },
        "rooms": rooms, "nodes": nodes, "graphs": graphs
    });
    let mut writer = BufWriter::new(
        File::create(&args.output)
            .with_context(|| format!("Creating {}", args.output.display()))?,
    );
    serde_json::to_writer(&mut writer, &output)?;
    writer.flush()?;
    Ok(())
}
