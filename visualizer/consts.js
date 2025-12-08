// Fractional tile adjustments for where item icons are drawn, keyed by 'room_id:node_id':
const offsets = {
	"46:3": [-.25, 0],    // Brinstar Reserve Tank Room: Middle Visible Item
	"46:4": [0.25, 0],    // Brinstar Reserve Tank Room: Right Hidden Item
	"43:3": [-0.25, 0],   // Billy Mays Room: Hidden Item
	"43:2": [0.25, 0],    // Billy Mays Room: Pedestal Item
	"99:3": [0.5, 0],     // Norfair Reserve Tank Room: Hidden Platform Item
	"181:2": [-0.25, 0],  // Watering Hole: Left Item
	"181:3": [0.25, 0],   // Watering Hole: Right Item
	"209:3": [-0.25, 0],  // West Sand Hole: Left Item
	"209:4": [0.25, 0],   // West Sand Hole: Right Item
	"21:5": [-0.25, 0],   // Green Pirates Shaft: Left Item
	"21:6": [0.25, 0],    // Green Pirates Shaft: Right Item
};
let item_plm = {
	"ETank": 0,
	"Missile": 1,
	"Super": 2,
	"PowerBomb": 3,
	"Bombs": 4,
	"Charge": 5,
	"Ice": 6,
	"HiJump": 7,
	"SpeedBooster": 8,
	"Wave": 9,
	"Spazer": 10,
	"SpringBall": 11,
	"Varia": 12,
	"Gravity": 13,
	"XRayScope": 14,
	"Plasma": 15,
	"Grapple": 16,
	"SpaceJump": 17,
	"ScrewAttack": 18,
	"Morph": 19,
	"ReserveTank": 20,
	"WallJump": 21,
	"None": 22,
	"Hidden": 23,
};
let door_enum = {
	"red":0,
	"green":1,
	"yellow":2,
	"charge":3,
	"ice":4,
	"spazer":5,
	"wave":6,
	"plasma":7,
	"blue":8,
};
let itemtypes = {
	"majors": ["Varia",
			"Gravity",
			"Morph",
			"SpaceJump",
			"ScrewAttack",
			"WallJump",
			"Bombs",
			"HiJump",
			"SpeedBooster",
			"SpringBall",
			"Grapple",
			"XRayScope",
			"Charge",
			"Ice",
			"Wave",
			"Spazer",
			"Plasma"],
	"health": ["ETank", "ReserveTank"],
	"ammo": ["Super","PowerBomb"],
	"missiles": ["Missile"]
}
let item_rank = {
	"Varia": 1,
	"Gravity": 2,
	"Morph": 3,
	"SpaceJump": 4,
	"ScrewAttack": 5,
	"WallJump": 6,
	"Bombs": 7,
	"HiJump": 8,
	"SpeedBooster": 9,
	"SpringBall": 10,
	"Grapple": 11,
	"XRayScope": 12,
	"Charge": 13,
	"Ice": 14,
	"Wave": 15,
	"Spazer": 16,
	"Plasma": 17,
	"ETank": 18,
	"ReserveTank": 19,
	"Super": 20,
	"PowerBomb": 21,
	"Missile": 22,
};

// Keyed by room ID:
let roomFlags = {
	"19": ["f_DefeatedBombTorizo", "Defeat Bomb Torizo", 0,0],
	"185": ["f_DefeatedBotwoon", "Defeat Botwoon",0.5,0],
	"122": ["f_DefeatedCrocomire", "Defeat Crocomire",4,0],
	"193": ["f_DefeatedDraygon", "Defeat Draygon",0.5,0.5],
	"150": ["f_DefeatedGoldenTorizo", "Defeat Golden Torizo",0.5,1],
	"84": ["f_DefeatedKraid", "Defeat Kraid",0.5,0.5],
	"238": ["f_DefeatedMotherBrain", "Defeat Mother Brain",1.5,0],
	"158": ["f_DefeatedPhantoon", "Defeat Phantoon",0,0],
	"142": ["f_DefeatedRidley", "Defeat Ridley",0,0.5],
	"57": ["f_DefeatedSporeSpawn", "Defeat Spore Spawn",0,1.5],
	"226": ["f_KilledMetroidRoom1", "Clear Metroid Room 1",2.5,0],
	"227": ["f_KilledMetroidRoom2", "Clear Metroid Room 2",0,0.5],
	"228": ["f_KilledMetroidRoom3", "Clear Metroid Room 3",2.5,0],
	"229": ["f_KilledMetroidRoom4", "Clear Metroid Room 4",0,0.5],
	"170": ["f_MaridiaTubeBroken", "Break Maridia Tube",0,1],
	"222": ["f_ShaktoolDoneDigging", "Clear Shaktool Room",1.5,0],
	"149": ["f_UsedAcidChozoStatue", "Use Acid Statue",0,0],
	"161": ["f_UsedBowlingStatue", "Use Bowling Statue",4,1],
	"12": ["f_ClearedPitRoom", "Clear Pit Room",1,0],
	"82": ["f_ClearedBabyKraidRoom", "Clear Baby Kraid Room",2.5,0],
	"219": ["f_ClearedPlasmaRoom", "Clear Plasma Room",0.5,1],
	"139": ["f_ClearedMetalPiratesRoom", "Clear Metal Pirates Room",1,0],
};
let flagtypes = {
	"bosses":["f_DefeatedKraid","f_DefeatedPhantoon",
		"f_DefeatedDraygon","f_DefeatedRidley","f_DefeatedMotherBrain"],
	"minibosses":["f_DefeatedBombTorizo","f_DefeatedSporeSpawn","f_DefeatedCrocomire",
		"f_DefeatedBotwoon","f_DefeatedGoldenTorizo"],
	"misc":["f_KilledMetroidRoom1","f_KilledMetroidRoom2","f_KilledMetroidRoom3",
		"f_KilledMetroidRoom4","f_MaridiaTubeBroken","f_ShaktoolDoneDigging",
		"f_UsedAcidChozoStatue","f_UsedBowlingStatue","f_ClearedPitRoom",
		"f_ClearedBabyKraidRoom","f_ClearedPlasmaRoom","f_ClearedMetalPiratesRoom",
		"f_ZebesAwake"]
}
let zebesawake = {
	"Baby Kraid Room": "f_ClearedBabyKraidRoom",
	"Plasma Room": "f_ClearedPlasmaRoom",
	"Metal Pirates Room": "f_ClearedMetalPiratesRoom",
	"Pit Room": "f_ClearedPitRoom",
}
let diff_colors = ["#00ff00","#ffff00","#ff0000","#ff8000","#00ffff", "#00ffff","#0080ff", "#0080ff","#ff00ff", "#ff00ff", "#ff0080"];