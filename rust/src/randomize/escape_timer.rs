use hashbrown::HashMap;
use ndarray::Array2;
use pathfinding;
use serde_derive::Deserialize;
use serde_derive::Serialize;

use crate::game_data::DoorPtrPair;
use crate::game_data::GameData;
use crate::game_data::IndexedVec;
use crate::game_data::Map;
use crate::game_data::RoomGeometry;
use crate::game_data::RoomGeometryDoor;
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
    pub distance: usize,
}

#[derive(Serialize, Deserialize)]
pub struct SpoilerEscape {
    pub distance: usize,
    pub time_per_distance: f32,
    pub difficulty_multiplier: f32,
    pub raw_time_seconds: f32,
    pub final_time_seconds: f32,
    pub animals_route: Option<Vec<SpoilerEscapeRouteEntry>>,
    pub ship_route: Vec<SpoilerEscapeRouteEntry>,
}

fn get_part_adjacency_matrix(room: &RoomGeometry) -> Array2<bool> {
    let num_parts = room.parts.len();
    let mut adj: Array2<u32> = Array2::zeros([num_parts, num_parts]);
    for i in 0..num_parts {
        adj[[i, i]] = 1;
    }
    for &(src_part, dst_part) in &room.durable_part_connections {
        adj[[src_part, dst_part]] = 1;
        adj[[dst_part, src_part]] = 1;
    }
    for &(src_part, dst_part) in &room.transient_part_connections {
        adj[[src_part, dst_part]] = 1;
    }
    adj = adj.dot(&adj);
    adj = adj.dot(&adj);
    let adj_bool: Array2<bool> = adj.mapv(|x| x > 0);
    adj_bool
}

fn get_part(door_idx: usize, room: &RoomGeometry) -> usize {
    for (i, part) in room.parts.iter().enumerate() {
        if part.contains(&door_idx) {
            return i;
        }
    }
    panic!(
        "Part for door_idx={door_idx} not found in room: {0}",
        room.name
    );
}

fn room_has_edge(
    src_door_idx: usize,
    dst_door_idx: usize,
    room: &RoomGeometry,
    part_adjacency: &Array2<bool>,
) -> bool {
    let src_part = get_part(src_door_idx, room);
    let dst_part = get_part(dst_door_idx, room);
    part_adjacency[[src_part, dst_part]]
}

fn get_xy(door: &RoomGeometryDoor) -> (f32, f32) {
    if door.direction == "left" {
        (door.x as f32, door.y as f32 + 0.5)
    } else if door.direction == "right" {
        (door.x as f32 + 1.0, door.y as f32 + 0.5)
    } else if door.direction == "up" {
        (door.x as f32 + 0.5, door.y as f32)
    } else if door.direction == "down" {
        (door.x as f32 + 0.5, door.y as f32 + 1.0)
    } else {
        panic!("Unexpected door direction: {}", door.direction);
    }
}

fn get_cost(src_door_idx: usize, dst_door_idx: usize, room: &RoomGeometry) -> Cost {
    let (src_x, src_y) = get_xy(&room.doors[src_door_idx]);
    let (dst_x, dst_y) = get_xy(&room.doors[dst_door_idx]);
    let distance = f32::abs(dst_x - src_x) + f32::abs(dst_y - src_y);
    distance as Cost
}

type OverridesMap = HashMap<RoomName, HashMap<(RoomGeometryDoorIdx, RoomGeometryDoorIdx), Cost>>;

fn get_overrides() -> OverridesMap {
    let overrides: Vec<(
        RoomName,
        Vec<((RoomGeometryDoorIdx, RoomGeometryDoorIdx), Cost)>,
    )> = vec![
        (
            "Terminator Room",
            vec![((0, 1), 3)],
        ),
        ("Tourian Escape Room 3", vec![((1, 0), 12), ((0, 1), 5)]),
        ("Tourian Escape Room 4", vec![((0, 1), 18)]),
        (
            "Parlor and Alcatraz",
            vec![((4, 1), 7), ((4, 2), 8), ((4, 5), 8), ((4, 6), 9)],
        ),
        (
            "West Ocean",
            vec![
                ((2, 3), 2),
                ((2, 4), 10),
                ((2, 6), 16),
                ((2, 7), 19),
                ((3, 4), 9),
                ((3, 6), 15),
                ((3, 7), 18),
                ((4, 6), 9),
                ((4, 7), 12),
            ],
        ),
        (
            "Green Brinstar Main Shaft",
            vec![
                ((4, 5), 10),
                // Shorten distances since timer is paused during elevator:
                ((9, 0), 2),
                ((9, 1), 3),
                ((9, 2), 4),
                ((9, 4), 8),
                ((9, 5), 11),
                ((9, 6), 2),
                ((9, 7), 4),
                ((9, 8), 5),
            ],
        ),
        (
            "Business Center",
            vec![
                ((6, 0), 1),
                ((6, 1), 2),
                ((6, 2), 3),
                ((6, 3), 1),
                ((6, 4), 3),
                ((6, 5), 4),
            ],
        ),
        (
            "Morph Ball Room",
            vec![
                // Shorten distances since timer is paused during elevator:
                ((2, 0), 6),
                ((2, 1), 2),
            ],
        ),
        (
            "Tourian First Room",
            vec![
                // Shorten distances since timer is paused during elevator:
                ((2, 0), 2),
                ((2, 1), 2),
            ],
        ),
        (
            "Caterpillar Room",
            vec![
                // Shorten distances since timer is paused during elevator:
                ((5, 0), 1),
                ((5, 1), 3),
                ((5, 2), 5),
                ((5, 3), 3),
                ((5, 4), 2),
            ],
        ),
        (
            "Maridia Elevator Room",
            vec![
                // Shorten distances since timer is paused during elevator:
                ((2, 0), 3),
                ((2, 1), 2),
            ],
        ),
        (
            "Main Hall",
            vec![
                ((2, 0), 4),
                ((2, 1), 3),
            ],
        ),
        (
            // Shorten effective distance since high speed is attainable
            "Frog Speedway",
            vec![
                ((0, 1), 3),
            ],
        ),
        (
            // Shorten effective distance since high speed is attainable
            "Speed Booster Hall",
            vec![
                ((0, 1), 4),
            ]
        ),
        (
            // Shorten effective distance since high speed is attainable
            "Crocomire Speedway",
            vec![
                ((0, 1), 5),
            ],
        ),
        ("Forgotten Highway Kago Room", vec![((0, 1), 6)]),
        (
            "Green Pirates Shaft",
            vec![
                ((1, 2), 5),
                ((1, 3), 10),
                ((1, 0), 11),
                ((2, 3), 6),
                ((2, 0), 7),
            ],
        ),
        (
            "Big Pink",
            vec![
                ((0, 1), 6),
                ((0, 2), 9),
                ((0, 3), 14),
                ((0, 4), 18),
                ((0, 6), 9),
                ((0, 7), 11),
                ((0, 8), 13),
                ((1, 2), 4),
                ((1, 3), 7),
                ((1, 4), 13),
                ((1, 5), 6),
                ((1, 8), 8),
                ((2, 3), 7),
                ((2, 4), 13),
                ((2, 5), 10),
                ((2, 8), 7),
                ((3, 4), 9),
                ((3, 5), 13),
                ((3, 6), 5),
                ((3, 7), 5),
                ((4, 5), 18),
                ((4, 6), 10),
                ((4, 7), 10),
                ((4, 8), 8),
                ((5, 6), 9),
                ((5, 7), 11),
                ((5, 8), 13),
                ((6, 8), 5),
                ((7, 8), 5),
            ],
        ),
        ("Pink Brinstar Power Bomb Room", vec![((0, 1), 4)]),
        ("Construction Zone", vec![((0, 1), 2)]),
        ("Etecoon Energy Tank Room", vec![((2, 3), 7)]),
        ("Pink Brinstar Hopper Room", vec![((0, 1), 4)]),
        ("Below Spazer", vec![((1, 2), 4)]),
        ("Warehouse Zeela Room", vec![((0, 2), 3)]),
        ("Warehouse Kihunter Room", vec![((1, 0), 4)]),
        ("Kraid Eye Door Room", vec![((1, 2), 3)]),
        ("Cathedral Entrance", vec![((0, 1), 5)]),
        ("Volcano Room", vec![((0, 1), 5)]),
        ("Lava Dive Room", vec![((0, 1), 6)]),
        ("Upper Norfair Farming Room", vec![((0, 1), 3)]),
        ("Acid Statue Room", vec![((0, 1), 6)]),
        ("Amphitheatre", vec![((0, 1), 8)]),
        (
            "Red Kihunter Shaft",
            vec![
                ((0, 2), 6),
                ((0, 3), 10),
                ((1, 2), 6),
                ((1, 3), 10),
                ((2, 3), 5),
            ],
        ),
        ("Three Musketeers' Room", vec![((0, 1), 8)]),
        ("Lower Norfair Fireflea Room", vec![((1, 2), 6)]),
        ("Bowling Alley", vec![((1, 2), 6)]),
        (
            "Wrecked Ship Main Shaft",
            vec![
                ((0, 1), 7),
                ((0, 3), 3),
                ((0, 4), 9),
                ((0, 5), 10),
                ((0, 6), 8),
                ((1, 2), 7),
                ((1, 3), 5),
                ((1, 5), 3),
                ((1, 6), 15),
                ((2, 3), 3),
                ((2, 4), 9),
                ((2, 5), 10),
                ((2, 6), 7),
                ((3, 4), 7),
                ((3, 5), 8),
                ((3, 6), 10),
                ((4, 5), 5),
                ((4, 6), 10),
                ((5, 6), 18),
            ],
        ),
        (
            "Electric Death Room",
            vec![((0, 2), 3), ((0, 1), 6), ((1, 2), 4)],
        ),
        ("Fish Tank", vec![((1, 0), 8), ((1, 3), 7)]),
        ("Crab Hole", vec![((0, 1), 2), ((2, 3), 2)]),
        (
            "Plasma Spark Room",
            vec![
                ((0, 1), 4),
                ((0, 2), 8),
                ((0, 3), 7),
                ((1, 2), 7),
                ((1, 3), 7),
                ((2, 3), 7),
            ],
        ),
        (
            "Aqueduct",
            vec![
                ((3, 7), 0), // Escape timer stops during toilet
                ((0, 1), 2),
                ((0, 2), 8),
                ((0, 4), 11),
                ((0, 5), 9),
                ((0, 6), 6),
                ((6, 0), 3), // asymmetric
                ((1, 4), 10),
                ((1, 5), 8),
                ((1, 6), 5),
                ((2, 4), 9),
                ((2, 5), 7),
                ((6, 4), 12),
                ((6, 5), 10),
            ],
        ),
        ("Pants Room", vec![((0, 2), 8), ((2, 0), 3)]),
        ("East Cactus Alley Room", vec![((0, 1), 8)]),
        ("Metroid Room 4", vec![((0, 1), 3)]),
        ("Tourian Escape Room 4", vec![((0, 1), 17)]),
        (
            "Single Chamber",
            vec![
                ((0, 1), 7),
                ((0, 2), 3),
                ((0, 3), 5),
                ((0, 4), 7),
                ((1, 2), 9),
                ((1, 3), 11),
                ((1, 4), 13),
                ((2, 3), 3),
                ((2, 4), 5),
                ((3, 4), 3),
            ],
        ),
    ];
    overrides
        .into_iter()
        .map(|(name, v)| (name, HashMap::from_iter(v.into_iter())))
        .collect()
}

fn get_override_cost(
    src_door_idx: usize,
    dst_door_idx: usize,
    room: &RoomGeometry,
    overrides: &OverridesMap,
) -> Option<Cost> {
    if overrides.contains_key(room.name.as_str()) {
        let room_overrides = &overrides[room.name.as_str()];
        if room_overrides.contains_key(&(src_door_idx, dst_door_idx)) {
            return Some(room_overrides[&(src_door_idx, dst_door_idx)]);
        } else if room_overrides.contains_key(&(dst_door_idx, src_door_idx)) {
            return Some(room_overrides[&(dst_door_idx, src_door_idx)]);
        }
    }
    None
}

// TODO: Get rid of the dependence on GameData since it is a bit circular.
pub fn get_base_room_door_graph(game_data: &GameData) -> RoomDoorGraph {
    let room_geometry = &game_data.room_geometry;
    let overrides = get_overrides();
    let mut vertices: IndexedVec<VertexKey> = IndexedVec::default();
    let mut successors: Vec<Vec<(VertexId, Cost)>> = Vec::new();
    for (room_idx, mut room) in room_geometry.iter().enumerate() {
        let mut new_room: RoomGeometry;
        if room.name == "Landing Site" {
            // Add fake "door" data for the Ship, to use as the escape destination.
            new_room = room.clone();
            let door_idx = new_room.doors.len();
            new_room.doors.push(RoomGeometryDoor {
                direction: "up".to_string(),
                x: 4,
                y: 4,
                exit_ptr: None,
                entrance_ptr: None,
                subtype: "normal".to_string(),
            });
            new_room.parts[0].push(door_idx);
            room = &new_room;
        }
        for (door_idx, _) in room.doors.iter().enumerate() {
            vertices.add(&(room_idx, door_idx));
            successors.push(vec![]);
        }
        let part_adjacency = get_part_adjacency_matrix(room);
        for src_door_idx in 0..room.doors.len() {
            for dst_door_idx in 0..room.doors.len() {
                if !room_has_edge(src_door_idx, dst_door_idx, room, &part_adjacency) {
                    continue;
                }
                let mut cost = get_cost(src_door_idx, dst_door_idx, room);
                if let Some(override_cost) =
                    get_override_cost(src_door_idx, dst_door_idx, room, &overrides)
                {
                    cost = override_cost;
                }
                let src_vertex_id = vertices.index_by_key[&(room_idx, src_door_idx)];
                let dst_vertex_id = vertices.index_by_key[&(room_idx, dst_door_idx)];
                successors[src_vertex_id].push((dst_vertex_id, cost));
            }
        }
    }

    let ship_room_idx = game_data.room_idx_by_name["Landing Site"];
    let ship_door_idx = room_geometry[ship_room_idx].doors.len();
    let ship_vertex_id = vertices.index_by_key[&(ship_room_idx, ship_door_idx)];

    let animals_room_door_idx =
        game_data.room_and_door_idxs_by_door_ptr_pair[&(Some(0x18BAA), Some(0x18BC2))];
    let animals_vertex_id = vertices.index_by_key[&animals_room_door_idx];

    let mb_room_door_idx =
        game_data.room_and_door_idxs_by_door_ptr_pair[&(Some(0x1AA8C), Some(0x1AAE0))];
    let mother_brain_vertex_id = vertices.index_by_key[&mb_room_door_idx];

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
                distance: dst_cost,
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
    let distance: usize;
    if difficulty.save_animals {
        let animals_path = get_shortest_path(
            graph.mother_brain_vertex_id,
            graph.animals_vertex_id,
            &graph,
        );
        animals_spoiler = Some(get_spoiler_escape_route(&animals_path, &graph, &game_data));
        let ship_path = get_shortest_path(graph.animals_vertex_id, graph.ship_vertex_id, &graph);
        ship_spoiler = get_spoiler_escape_route(&ship_path, &graph, &game_data);
        distance = animals_path.last().unwrap().1 + ship_path.last().unwrap().1;
    } else {
        animals_spoiler = None;
        let ship_path =
            get_shortest_path(graph.mother_brain_vertex_id, graph.ship_vertex_id, &graph);
        ship_spoiler = get_spoiler_escape_route(&ship_path, &graph, &game_data);
        distance = ship_path.last().unwrap().1;
    }

    let time_per_distance: f32 = 1.6;
    let raw_time_seconds = distance as f32 * time_per_distance * difficulty.escape_timer_multiplier;
    let mut final_time_seconds = f32::ceil(raw_time_seconds / 5.0) * 5.0;
    if final_time_seconds > 5995.0 {
        final_time_seconds = 5995.0;
    }

    SpoilerEscape {
        distance: distance,
        time_per_distance: time_per_distance,
        difficulty_multiplier: difficulty.escape_timer_multiplier,
        raw_time_seconds: raw_time_seconds,
        final_time_seconds: final_time_seconds,
        animals_route: animals_spoiler,
        ship_route: ship_spoiler,
    }
}
