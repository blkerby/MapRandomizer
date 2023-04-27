use hashbrown::HashMap;
use pathfinding;
use serde_derive::Deserialize;
use serde_derive::Serialize;

use crate::game_data::DoorPtrPair;
use crate::game_data::EscapeTimingRoom;
use crate::game_data::GameData;
use crate::game_data::IndexedVec;
use crate::game_data::Map;
use crate::game_data::RoomGeometryDoorIdx;
use crate::game_data::RoomGeometryRoomIdx;

use super::DifficultyConfig;

pub type RoomName = &'static str;
pub type VertexId = usize;
pub type Cost = usize;
pub type VertexKey = (RoomGeometryRoomIdx, RoomGeometryDoorIdx);

#[derive(Clone, Default)]
pub struct RoomDoorGraph {
    pub vertices: IndexedVec<VertexKey>,
    pub successors: Vec<Vec<(VertexId, Cost)>>,
    pub mother_brain_vertex_id: VertexId,
    pub ship_vertex_id: VertexId,
    pub animals_vertex_id: VertexId,
}

// Ideally this would contain coords, but whatever
#[derive(Serialize, Deserialize)]
pub struct SpoilerEscapeRouteNode {
    room: String,
    node: String,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerEscapeRouteEntry {
    pub from: SpoilerEscapeRouteNode,
    pub to: SpoilerEscapeRouteNode,
    pub base_igt_frames: usize,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerEscape {
    pub base_igt_frames: usize,
    pub base_igt_seconds: f32,
    pub base_leniency_factor: f32,
    pub difficulty_multiplier: f32,
    pub raw_time_seconds: f32,
    pub final_time_seconds: f32,
    pub animals_route: Option<Vec<SpoilerEscapeRouteEntry>>,
    pub ship_route: Vec<SpoilerEscapeRouteEntry>,
}

// In game times are recorded in the format X.YY where X is integer seconds and YY is frames 
// (as in the Practice Hack). This function converts this to frames.
fn parse_in_game_time(raw_time: f32) -> usize {
    let int_part = raw_time.floor() as usize;
    let frac_part = raw_time.fract();
    let frames = (frac_part * 100.0) as usize;
    assert!(frames < 60);
    int_part * 60 + frames
}

pub fn get_base_room_door_graph(escape_timings: &[EscapeTimingRoom]) -> RoomDoorGraph {
    let mut vertices: IndexedVec<VertexKey> = IndexedVec::default();
    let mut successors: Vec<Vec<(VertexId, Cost)>> = Vec::new();
    let mut ship_vertex_id = VertexId::MAX;
    let mut animals_vertex_id = VertexId::MAX;
    let mut mother_brain_vertex_id = VertexId::MAX;

    for (room_idx, escape_timing_room) in escape_timings.iter().enumerate() {
        for escape_timing_group in &escape_timing_room.timings {
            let from_key = (room_idx, escape_timing_group.from_door.door_idx);
            let from_idx = vertices.add(&from_key);
            successors.push(vec![]);
            if escape_timing_room.room_name == "Landing Site" && escape_timing_group.from_door.name == "Ship" {
                ship_vertex_id = from_idx;
            }
            if escape_timing_room.room_name == "Bomb Torizo Room" {
                animals_vertex_id = from_idx;
            }
            if escape_timing_room.room_name == "Mother Brain Room" && escape_timing_group.from_door.direction == "left" {
                mother_brain_vertex_id = from_idx;
            }
        }
        for escape_timing_group in &escape_timing_room.timings {
            let from_key = (room_idx, escape_timing_group.from_door.door_idx);
            let from_idx = vertices.index_by_key[&from_key];
            for escape_timing in &escape_timing_group.to {
                let to_key = (room_idx, escape_timing.to_door.door_idx);
                let to_idx = vertices.index_by_key[&to_key];
                if let Some(in_game_time) = escape_timing.in_game_time {
                    let cost = parse_in_game_time(in_game_time);
                    successors[from_idx].push((to_idx, cost));
                }
            }
        }
    }

    RoomDoorGraph {
        vertices,
        successors,
        mother_brain_vertex_id,
        ship_vertex_id,
        animals_vertex_id,
    }
}

pub fn get_full_room_door_graph(game_data: &GameData, map: &Map) -> RoomDoorGraph {
    let base = &game_data.base_room_door_graph;
    let mut door_ptr_pair_to_vertex: HashMap<DoorPtrPair, VertexId> = HashMap::new();
    for (room_idx, room) in game_data.room_geometry.iter().enumerate() {
        println!("{}", room.name);
        for (door_idx, door) in room.doors.iter().enumerate() {
            let vertex_id = base.vertices.index_by_key[&(room_idx, door_idx)];
            let door_ptr_pair = (door.exit_ptr, door.entrance_ptr);
            door_ptr_pair_to_vertex.insert(door_ptr_pair, vertex_id);
        }
    }
    let mut room_door_graph = base.clone();
    for &(src_door_ptr_pair, dst_door_ptr_pair, bidirectional) in &map.doors {
        let src_vertex_id = door_ptr_pair_to_vertex[&src_door_ptr_pair];
        let dst_vertex_id = door_ptr_pair_to_vertex[&dst_door_ptr_pair];
        room_door_graph.successors[src_vertex_id].push((dst_vertex_id, 0));
        if bidirectional {
            room_door_graph.successors[dst_vertex_id].push((src_vertex_id, 0));
        }
    }
    room_door_graph
}

fn get_vertex_name(
    vertex_id: VertexId,
    room_door_graph: &RoomDoorGraph,
    game_data: &GameData,
) -> SpoilerEscapeRouteNode {
    if vertex_id == room_door_graph.ship_vertex_id {
        return SpoilerEscapeRouteNode {
            room: "Landing Site".to_string(),
            node: "Ship".to_string(),
        };
    }
    let (room_idx, door_idx) = room_door_graph.vertices.keys[vertex_id];
    let door = &game_data.room_geometry[room_idx].doors[door_idx];
    let door_ptr_pair = (door.exit_ptr, door.entrance_ptr);
    let (room_id, door_id) = game_data.door_ptr_pair_map[&door_ptr_pair];
    let room_json = &game_data.room_json_map[&room_id];
    let node_json = &game_data.node_json_map[&(room_id, door_id)];
    SpoilerEscapeRouteNode {
        room: room_json["name"].as_str().unwrap().to_string(),
        node: node_json["name"].as_str().unwrap().to_string()
    }
}

fn get_spoiler_escape_route(
    path: &[(VertexId, Cost)],
    room_door_graph: &RoomDoorGraph,
    game_data: &GameData,
) -> Vec<SpoilerEscapeRouteEntry> {
    let mut out: Vec<SpoilerEscapeRouteEntry> = Vec::new();
    for slice in path.windows(2) {
        if let &[(src_vertex_id, src_cost), (dst_vertex_id, dst_cost)] = slice {
            if src_cost == dst_cost {
                continue;
            }
            out.push(SpoilerEscapeRouteEntry {
                from: get_vertex_name(src_vertex_id, room_door_graph, game_data),
                to: get_vertex_name(dst_vertex_id, room_door_graph, game_data),
                base_igt_frames: dst_cost,
            });
        } else {
            panic!("Internal error");
        }
    }
    out
}

fn get_shortest_path(
    src: VertexId,
    dst: VertexId,
    room_door_graph: &RoomDoorGraph,
) -> Vec<(VertexId, Cost)> {
    let successors = |&src: &VertexId| room_door_graph.successors[src].iter().copied();
    let parents: std::collections::HashMap<VertexId, (VertexId, Cost)> =
        pathfinding::directed::dijkstra::dijkstra_all(&src, successors);
    let mut path: Vec<(VertexId, Cost)> = Vec::new();
    let mut v = dst;
    while v != src {
        let (new_v, cost) = parents[&v];
        path.push((v, cost));
        v = new_v;
    }
    path.reverse();
    path
}

pub fn compute_escape_data(
    game_data: &GameData,
    map: &Map,
    difficulty: &DifficultyConfig,
) -> SpoilerEscape {
    let graph = get_full_room_door_graph(game_data, map);
    let animals_spoiler: Option<Vec<SpoilerEscapeRouteEntry>>;
    let ship_spoiler: Vec<SpoilerEscapeRouteEntry>;
    let base_igt_frames: usize;
    if difficulty.save_animals {
        let animals_path = get_shortest_path(
            graph.mother_brain_vertex_id,
            graph.animals_vertex_id,
            &graph,
        );
        animals_spoiler = Some(get_spoiler_escape_route(&animals_path, &graph, &game_data));
        let ship_path = get_shortest_path(graph.animals_vertex_id, graph.ship_vertex_id, &graph);
        ship_spoiler = get_spoiler_escape_route(&ship_path, &graph, &game_data);
        base_igt_frames = animals_path.last().unwrap().1 + ship_path.last().unwrap().1;
    } else {
        animals_spoiler = None;
        let ship_path =
            get_shortest_path(graph.mother_brain_vertex_id, graph.ship_vertex_id, &graph);
        ship_spoiler = get_spoiler_escape_route(&ship_path, &graph, &game_data);
        base_igt_frames = ship_path.last().unwrap().1;
    }

    let base_igt_seconds: f32 = (base_igt_frames as f32) / 60.0;
    let base_leniency_factor: f32 = 1.15;
    let raw_time_seconds = base_igt_seconds * base_leniency_factor * difficulty.escape_timer_multiplier;
    let mut final_time_seconds = f32::ceil(raw_time_seconds / 5.0) * 5.0;
    if final_time_seconds > 5995.0 {
        final_time_seconds = 5995.0;
    }

    SpoilerEscape {
        base_igt_frames,
        base_igt_seconds,
        base_leniency_factor,
        difficulty_multiplier: difficulty.escape_timer_multiplier,
        raw_time_seconds: raw_time_seconds,
        final_time_seconds: final_time_seconds,
        animals_route: animals_spoiler,
        ship_route: ship_spoiler,
    }
}
