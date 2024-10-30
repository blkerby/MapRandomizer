const offsets = {
	"Morph Ball Room: Left Item": [2, 2],
	"Morph Ball Room: Right Item": [4, 2],
	"Cathedral: Hidden Item": [2, 1],
	"The Precious Room: Top Right Hidden Item": [1, 0],
	"Mama Turtle Room: Hidden Right Item": [2, 1],
	"Mama Turtle Room: Top Item": [1, 0],
	"Brinstar Reserve Tank Room: Middle Visible Item": [0.75, 0],
	"Brinstar Reserve Tank Room: Right Hidden Item": [1.25, 0],
	"Brinstar Reserve Tank Room: Chozo Item": [0, 0],
	"Lower Norfair Fireflea Room: Firefleas Item": [2, 5],
	"Main Street: Morph Tunnel Item": [1, 2],
	"Main Street: Speed Blocked Item": [0, 3],
	"Gauntlet Energy Tank Room: Item": [5, 0],
	"Wrecked Ship Main Shaft: Item": [0, 5],
	"Pseudo Plasma Spark Room: Hidden Item": [2, 2],
	"Assembly Line: Item": [2, 0],
	"Warehouse Kihunter Room: Hidden Item": [2, 0],
	"Billy Mays Room: Pedestal Item": [0.25, 0],
	"Billy Mays Room: Hidden Item": [-0.25, 0],
	"Norfair Reserve Tank Room: Hidden Platform Item": [0.5, 0],
	"Green Hill Zone: Item (Through the Pipe)": [3, 1],
	"Crocomire's Room: Item": [7, 0],
	"Blue Brinstar Energy Tank Room: Hidden Ceiling Item": [1, 2],
	"Blue Brinstar Energy Tank Room: Right Item": [2, 2],
	"Three Musketeers' Room: Hidden Item": [0, 2],
	"Alpha Power Bomb Room: Chozo Item": [1, 0],
	"Alpha Power Bomb Room: Hidden Left Item": [0, 0],
	"Aqueduct: Top Right Right Item": [5, 0],
	"Aqueduct: Top Right Left Item": [4, 0],
	"Screw Attack Room: Item": [0, 2],
	"Bubble Mountain: Bottom Right Item": [1, 3],
	"Watering Hole: Left Item": [-0.25, 2],
	"Watering Hole: Right Item": [0.25, 2],
	"Speed Booster Hall: Hidden Item": [11, 1],
	"West Sand Hole: Top Left - Right Item": [0.25, 0],
	"West Sand Hole: Top Left - Left Item": [-0.25, 0],
	"East Sand Hole: Top Left Item": [0, 0],
	"East Sand Hole: Right Item": [1, 1],
	"Post Crocomire Missile Room: Item": [3, 0],
	"Double Chamber: Item": [1, 0],
	"Beta Power Bomb Room: Item": [0, 1],
	"West Ocean: Bottom Left Underwater Item": [0, 5],
	"West Ocean: Left Morph Maze Item": [0, 2],
	"West Ocean: Top Hidden Item (Above Trippers)": [1, 0],
	"Early Supers Room: Lower Item": [1, 1],
	"Early Supers Room: Top Left Item": [0, 0],
	"Plasma Room: Item": [1, 2],
	"Green Pirates Shaft: Left Item": [-0.25, 1],
	"Green Pirates Shaft: Right Item": [0.25, 1],
	"Spring Ball Room: Item": [1, 1],
	"Pink Brinstar Power Bomb Room: Item": [0, 1],
	"Pit Room: Item": [0, 1],
	"Mickey Mouse Room: Item": [2, 1],
	"Grapple Beam Room: Item": [0, 2],
	"Golden Torizo's Room: Hidden Right Item": [1, 0],
	"Golden Torizo's Room: Top Left Item": [0, 0],
	"Wrecked Ship East Super Room: Item (Behind the Bomb Wall)": [3, 0],
	"Botwoon Energy Tank Room: Item": [3, 0],
	"Terminator Room: Item": [0, 2],
	"Post Crocomire Jump Room: Item": [4, 0],
	"Green Bubbles Missile Room: Item": [1, 0],
	"Lower Norfair Spring Ball Maze Room: Item": [2, 0],
	"Crateria Power Bomb Room: Item": [1, 0],
	"Green Brinstar Main Shaft: Right Etecoon Shaft - Item": [3, 7],
	"Crateria Super Room: Item": [3, 0],
	"Spore Spawn Super Room: Item": [1, 8],
	"Hi Jump Energy Tank Room: Top Right Item": [1, 0],
	"Hi Jump Energy Tank Room: Top Left Item": [0, 0],
	"Big Pink: Top Item (Above Hopper Pit)": [2, 3],
	"Big Pink: Middle Item": [2, 6],
	"Big Pink: Bottom Chozo Item": [2, 7],
	"Bowling Alley: Bottom Chozo Item": [3, 2],
	"Bowling Alley: Top Right Item": [5, 0],
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
};
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
}
let roomFlags = {
	"Bomb Torizo Room": ["f_DefeatedBombTorizo", "Defeat Bomb Torizo"],
	"Botwoon's Room": ["f_DefeatedBotwoon", "Defeat Botwoon"],
	"Crocomire's Room": ["f_DefeatedCrocomire", "Defeat Crocomire"],
	"Draygon's Room": ["f_DefeatedDraygon", "Defeat Draygon"],
	"Golden Torizo's Room": ["f_DefeatedGoldenTorizo", "Defeat Golden Torizo"],
	"Kraid Room": ["f_DefeatedKraid", "Defeat Kraid"],
	"Mother Brain Room": ["f_DefeatedMotherBrain", "Defeat Mother Brain"],
	"Phantoon's Room": ["f_DefeatedPhantoon", "Defeat Phantoon"],
	"Ridley's Room": ["f_DefeatedRidley", "Defeat Ridley"],
	"Spore Spawn Room": ["f_DefeatedSporeSpawn", "Defeat Spore Spawn"],
	"Metroid Room 1": ["f_KilledMetroidRoom1", "Clear Metroid Room 1"],
	"Metroid Room 2": ["f_KilledMetroidRoom2", "Clear Metroid Room 2"],
	"Metroid Room 3": ["f_KilledMetroidRoom3", "Clear Metroid Room 3"],
	"Metroid Room 4": ["f_KilledMetroidRoom4", "Clear Metroid Room 4"],
	"Glass Tunnel": ["f_MaridiaTubeBroken", "Break Maridia Tube"],
	"Shaktool Room": ["f_ShaktoolDoneDigging", "Clear Shaktool Room"],
	"Acid Statue Room": ["f_UsedAcidChozoStatue", "Use Acid Statue"],
	"Bowling Alley": ["f_UsedBowlingStatue", "Use Bowling Statue"],
	"Pit Room": ["f_ClearedPitRoom", "Clear Pit Room"],
	"Baby Kraid Room": ["f_ClearedBabyKraidRoom", "Clear Baby Kraid Room"],
	"Plasma Room": ["f_ClearedPlasmaRoom", "Clear Plasma Room"],
	"Metal Pirates Room": ["f_ClearedMetalPiratesRoom", "Clear Metal Pirates Room"],
	"Parlor And Alcatraz": ["f_ZebesAwake", "Awaken Zebes"],
	"Climb": ["f_ZebesAwake", "Awaken Zebes"],
	"Construction Zone": ["f_ZebesAwake", "Awaken Zebes"],
	"The Final Missile": ["f_ZebesAwake", "Awaken Zebes"],
	"Morph Ball Room": ["f_ZebesAwake", "Awaken Zebes"],
	"Blue Brinstar Energy Tank Room": ["f_ZebesAwake", "Awaken Zebes"],
};

function lookupOffset(room, node) {
	key = room + ": " + node
	return offsets[key];
}

fetch(`../spoiler.json`).then(c => c.json()).then(c => {
	// generate map
	let map = new Array(72 * 72).fill(-1);
	for (let i in c.all_rooms) {
		let v = c.all_rooms[i];
		for (let y = 0; y < v.map.length; y++) {
			for (let x = 0; x < v.map[y].length; x++) {
				if (v.map[y][x] != 0) {
					map[(v.coords[1] + y) * 72 + (v.coords[0] + x)] = +i;
				}
			}
		}
	}
	let step_limit = null;
	let icon = id => {
		let el = document.createElement("span");
		el.className = "ui-icon";
		el.style.backgroundPositionX = `-${id * 16}px`;
		return el;
	}
	let update_selected = () => {
		if (document.getElementById(`step-null`) !== null) {
			document.getElementById(`step-null`).classList.remove("selected");
			for (let i of c.summary) {
				document.getElementById(`step-${i.step}`).classList.remove("selected");
			}
			document.getElementById(`step-${step_limit}`).classList.add("selected");
		}
	}
	let show_overview = () => {
		let si = document.getElementById("sidebar-info");
		si.innerHTML = "";
		let seen = new Set();
		for (let i in c.summary) {
			let step_div = document.createElement("div");
			step_div.id = `step-${c.summary[i].step}`;
			step_div.className = "step-panel";
			step_div.onclick = () => {
				gen_obscurity(c.summary[i].step);
			}

			let step_number = document.createElement("span");
			step_number.className = "step-number";
			if (i == c.summary.length - 1 && c.summary[i].items.length == 0) {
				step_number.className = "step-whole-map";
				step_number.innerHTML = `FINAL MAP`;
			} else {
				step_number.innerHTML = `${c.summary[i].step}`;
			}
			step_div.appendChild(step_number);

			let items = c.details[i].items;
			let sortedItemIdxs = Array.from(items.keys()).sort((a, b) => item_rank[items[a].item] - item_rank[items[b].item]);
			let first = true;
			for (item_idx of sortedItemIdxs) {
				let j = items[item_idx];
				if (!seen.has(j.item)) {
					let el = icon(item_plm[j.item]);
					el.className = "ui-icon-hoverable";
					el.onclick = ev => {
						show_item_details(j.item, j.location, i, j);
					}
					step_div.appendChild(el);

					if (first) {
						step_div.ondblclick = () => {
							show_item_details(j.item, j.location, i, j);
						}
					}
					first = false;

					seen.add(j.item);
				}
			}
			if (first && sortedItemIdxs.length > 0) {
				let j = items[sortedItemIdxs[0]];
				step_div.ondblclick = () => {
					show_item_details(j.item, j.location, i, j);
				}
			}
			si.appendChild(step_div);
		}

		step_div = document.createElement("div");
		step_div.id = `step-null`;
		step_div.className = "step-panel";
		step_div.onclick = () => gen_obscurity(null);

		step_number = document.createElement("span");
		step_number.className = "step-whole-map";
		step_number.innerHTML = 'WHOLE MAP';
		help_button = document.createElement("i");
		help_button.className = "bi bi-question-circle-fill help-button";
		help_button.setAttribute("data-bs-theme", "dark");
		help_button.style = "margin-left:auto;";
		help_button.onclick = (e) => {
			document.getElementById("msg-wrap").style.display = "flex";
			e.stopPropagation();
		}
		// help_button.onmouseup = () => {
		// 	return true;
		// }
		// help_button.onclick = () => {
		// 	return false;
		// }
		step_div.appendChild(step_number);
		step_div.appendChild(help_button);

		si.appendChild(step_div);
		update_selected();
	}
	show_overview();
	window.gen_obscurity = (sl) => {
		step_limit = sl;
		update_selected();

		// generate obscurity overlay
		let ov = document.getElementById("obscure-overlay");
		let ctx = ov.getContext("2d");
		let img = ctx.createImageData(72, 72);

		if (step_limit === null) {
			ctx.putImageData(img, 0, 0);
			return;
		}
		for (let i = 0; i < 72 * 72; i++) {
			img.data[i * 4 + 3] = 0xD8; // mostly opaque
		}
		for (let v of c.all_rooms) {
			for (let y = 0; y < v.map.length; y++) {
				for (let x = 0; x < v.map[y].length; x++) {
					if (v.map[y][x] == 1) {
						let addr = (v.coords[1] + y) * 72 + (v.coords[0] + x);
						if (v.map_bireachable_step[y][x] < step_limit) {
							img.data[addr * 4 + 3] = 0x00; // transparent
						} else if (v.map_reachable_step[y][x] < step_limit) {
							img.data[addr * 4 + 3] = 0x7F; // semiopaque
						} else {
							img.data[addr * 4 + 3] = 0xD8; // mostly opaque
						}
					}
				}
			}
		}
		ctx.putImageData(img, 0, 0);
	}
	gen_obscurity(null);

	let el = document.getElementById("room-info");

	let dragged = false;
	var scale = 1, page_x = 0, page_y = 0;
	let m = document.getElementById("map");
	let evCache = [];
	let odist = -1;
	function transfo() {
		document.getElementById("zoom").style.transform =
		`translate(${page_x}px, ${page_y}px) scale(${scale})`;
	}
	function hover(ev) {
		let x = ((ev.offsetX / 24) | 0) - 1;
		let y = ((ev.offsetY / 24) | 0) - 1;
		if (x >= 0 && x < 72 && y >= 0 && y < 72) {
			let tile = map[y * 72 + x];
			if (tile >= 0) {
				el.innerText = c.all_rooms[tile].room;
				el.dataset.roomId = c.all_rooms[tile].room_id;
				el.style.left = ev.offsetX + 16 + "px";
				el.style.top = ev.offsetY + "px";
				el.classList.remove("hidden");
				return;
			}
		}
		el.classList.add("hidden")
		el.innerText = "";
	}
	function click() {
		if (el.innerText in roomFlags) {
			let flagPair = roomFlags[el.innerText];
			let flagName = flagPair[0];
			showFlag(c.details, flagName);
			if (el.innerText == "Mother Brain Room") {
				let path = "";
				for (let i of c.escape.animals_route ?? []) {
					for (let k of [i.from, i.to]) {
						let x = k.x * 24 + 24 + 12;
						let y = k.y * 24 + 24 + 12;
						path += `${path == "" ? "M" : "L"}${x} ${y} `;
					}
				}
				for (let i of c.escape.ship_route) {
					for (let k of [i.from, i.to]) {
						let x = k.x * 24 + 24 + 12;
						let y = k.y * 24 + 24 + 12;
						path += `${path == "" ? "M" : "L"}${x} ${y} `;
					}
				}
				document.getElementById("path-overlay").innerHTML += `<path d="${path}" stroke="black" fill="none" stroke-linejoin="round" stroke-width="4"/>`
				document.getElementById("path-overlay").innerHTML += `<path d="${path}" stroke="cyan" fill="none" stroke-linejoin="round" stroke-width="2"/>`
			}
		} else {
			console.log("bye")
			if (!dragged) {
				// deselect
				show_overview();
				document.getElementById("path-overlay").innerHTML = ""
			}
		}
	}
	function dblclick() {
		if (!el.classList.contains("hidden")) {
			window.open("/logic/room/" + el.dataset.roomId);
		}
	}
	function zm(x, y, delta) {
		const scaleOld = scale;
		var z = document.getElementById("zoom");

		scale *= 1.0 - delta * 0.0005;
		scale = Math.min(Math.max(0.25, scale), 100);

		var xorg = x - page_x - z.offsetWidth/2;
		var yorg = y - page_y - z.offsetHeight/2;

		var xnew = xorg / scaleOld;
		var ynew = yorg / scaleOld;
		
		xnew *= scale;
		ynew *= scale;

		var xdiff = xorg -xnew;
		var ydiff = yorg -ynew;

		page_x += xdiff;
		page_y += ydiff;

		transfo();
	}
	function up(ev) {
		ev.preventDefault();
		if (dragged)
			el.classList.add("hidden");
		
		evCache.splice(evCache.findIndex((cached) => cached.pointerID == ev.pointerID), 1)
		if (evCache.length != 1) {
			dragged = false;
		}
	}
	m.onpointerdown = ev => {
		if (ev.button != 0)
			return;

		ev.preventDefault();
		evCache.push(ev);
		dragged = false;
		if (evCache.length == 2) {
			let dx = Math.abs(evCache[0].x-evCache[1].x);
			let dy = Math.abs(evCache[0].y-evCache[1].y);
			odist = Math.sqrt(dx**2+dy**2);
		}
	}
	let fclick = true, timer = null;
	m.onpointerup = ev => {
		if (ev.button != 0)
			return;
		
		if (evCache.length == 1) {
			if (fclick) {
				click();
				timer = setTimeout(function (){
					fclick = true;
				}, 500);
				fclick = false;
			} else {	
				fclick = true;
				if (timer)
					clearTimeout(timer);
				let oldroom = el.innerText;
				hover(ev);
				if (oldroom == el.innerText)
					dblclick();
			}
		}
		up(ev);
	}
	document.body.onpointerleave = ev => {
		up(ev);
	}
	document.body.onpointerup = ev => {
		up(ev);
	}
	m.onpointermove = ev => {
		ev.preventDefault();
		if (evCache.length == 2) {
			var dx = Math.abs(evCache[0].x - evCache[1].x);
			var dy = Math.abs(evCache[0].y - evCache[1].y);
			var dist = Math.sqrt(dx**2 + dy**2);
			var delta = odist-dist;
			let i = evCache.findIndex((e) => e.pointerId == ev.pointerId);
			evCache[i] = ev;
			zm((evCache[0].x+evCache[1].x)/2, (evCache[0].y+evCache[1].y)/2,delta*2);
			odist = dist;
		} else if (evCache.length == 1) {
			dragged = true;
			page_x += ev.x - evCache[0].x;
			page_y += ev.y - evCache[0].y;
			evCache[0] = ev;
			transfo();
		} else if (evCache.length == 0) {
			// mouse only.
			hover(ev);
		}
	}
	m.onwheel = ev => {
		zm(ev.x, ev.y, ev.deltaY);
	}
	let createDiv = (html) => {
		const div = document.createElement('div');
		div.innerHTML = html;
		return div;
	};
	let createHtmlElement = (html) => {
		const template = document.createElement('template');
		template.innerHTML = html;
		return template.content.firstChild;
	};
	let show_item_details = (item_name, loc, i, j) => {
		if (j !== null) {
			let path = "";
			for (let k of j.return_route) {
				if (k.coords) {
					let x = k.coords[0] * 24 + 24 + 12;
					let y = k.coords[1] * 24 + 24 + 12;
					path += `${path == "" ? "M" : "L"}${x} ${y} `;
				}
			}
			document.getElementById("path-overlay").innerHTML = `<path d="${path}" stroke="black" fill="none" stroke-linejoin="round" stroke-width="4"/>`
			document.getElementById("path-overlay").innerHTML += `<path d="${path}" stroke="yellow" fill="none" stroke-linejoin="round" stroke-width="2"/>`
			path = "";
			for (let k of j.obtain_route) {
				if (k.coords) {
					let x = k.coords[0] * 24 + 24 + 12;
					let y = k.coords[1] * 24 + 24 + 12;
					path += `${path == "" ? "M" : "L"}${x} ${y} `;
				}
			}
			document.getElementById("path-overlay").innerHTML += `<path d="${path}" stroke="black" fill="none" stroke-linejoin="round" stroke-width="4"/>`
			document.getElementById("path-overlay").innerHTML += `<path d="${path}" stroke="white" fill="none" stroke-linejoin="round" stroke-width="2"/>`
		}
		let si = document.getElementById("sidebar-info");
		si.scrollTop = 0;
		si.innerHTML = "";
		if (j !== null) {
			step_limit = c.details[i].step;
			let title = document.createElement("div");
			title.className = "sidebar-title";
			title.innerHTML = `STEP ${step_limit}`;
			si.appendChild(title);
		}

		if (j !== null) {
			gen_obscurity(step_limit);

			let previous_header = document.createElement("div");
			previous_header.className = "category";
			previous_header.innerHTML = "PREVIOUSLY COLLECTIBLE";
			si.appendChild(previous_header);

			let ss = c.details[i].start_state;
			let s = [ss.max_missiles, ss.max_supers, ss.max_power_bombs, Math.floor(ss.max_energy / 100), ss.max_reserves / 100];
			let ic = [1, 2, 3, 0, 20];
			for (let i in s) {
				if (s[i] > 0) {
					si.appendChild(icon(ic[i]));
					let count = document.createElement("span");
					count.innerHTML = s[i] + " ";
					si.appendChild(count);
				}
			}
			for (let i of ss.items) {
				if (i == "Nothing") { continue; }
				if (!ic.includes(item_plm[i])) {
					si.appendChild(icon(item_plm[i]));
				}
			}

			let collectible_header = document.createElement("div");
			collectible_header.className = "category";
			collectible_header.innerHTML = "COLLECTIBLE ON THIS STEP";
			si.appendChild(collectible_header);

			let item_list = document.createElement("div");
			item_list.className = "item-list";
			let items = c.details[i].items;
			sortedItemIdxs = Array.from(items.keys()).sort((a, b) => item_rank[items[a].item] - item_rank[items[b].item]);
			for (item_idx of sortedItemIdxs) {
				let item = items[item_idx];
				let icon_el = icon(item_plm[item.item]);
				icon_el.className = "ui-icon-hoverable";
				icon_el.onclick = ev => {
					show_item_details(item.item, item.location, i, item);
				}
				if (item == j) {
					icon_el.classList.add("selected")
				}
				item_list.appendChild(icon_el);
			}
			si.appendChild(item_list);
		}

		let item_info = document.createElement("div");
		let item_difficulty = "";
		if (j !== null && j.difficulty !== null && j.difficulty !== undefined) {
			item_difficulty = ` (${j.difficulty})`
		}
		item_info.appendChild(createHtmlElement(`<div class="sidebar-item-name">${item_name}${item_difficulty}</div>`));
		item_info.appendChild(createHtmlElement(`<div class="category">LOCATION</div>`));
		item_info.appendChild(createDiv(`${loc.room}: ${loc.node}<br><small>${loc.area}</small>`));
		if (j !== null) {
			let ss = c.details[i].start_state;
			item_info.appendChild(createHtmlElement(`<div class="category">OBTAIN ROUTE</div>`));
			let lastRoom = null;
			let lastNode = null;
			for (let k of j.obtain_route) {
				let strat_url = `/logic/room/${k.room_id}/${k.from_node_id}/${k.to_node_id}/${k.strat_id}`;
				let nodeStr;
				if (k.strat_id !== null) {
					nodeStr = `<a style="text-decoration:none" href="${strat_url}">${k.room}: ${k.node}</a><br>`;
				} else {
					nodeStr = `${k.room}: ${k.node}<br>`;
				}
				if (k.room != lastRoom || k.node != lastNode || k.strat_id !== null) {
					item_info.appendChild(createDiv(nodeStr));
					lastRoom = k.room;
					lastNode = k.node;
				}
				let out = "";
				if (!k.strat_name.startsWith("Base ") && k.strat_name != "Base" && k.strat_name != "Leave Normally") {
					if (k.strat_notes) {
						let title = "";
						for (let i of k.strat_notes) {
							title += `${i} `;
						}
						out += `Strat: <a href=${strat_url}><abbr title="${title}">${k.strat_name}</abbr></a><br>`;
					} else {
						out += `Strat: <a href=${strat_url}>${k.strat_name}</a><br>`;
					}
				}
				if (k.energy_used !== undefined) {
					out += `Energy remaining: ${ss.max_energy - k.energy_used}<br>`;
				}
				if (k.reserves_used !== undefined) {
					out += `Reserves remaining: ${ss.max_reserves - k.reserves_used}<br>`;
				}
				if (k.missiles_used !== undefined) {
					out += `Missiles remaining: ${ss.max_missiles - k.missiles_used}<br>`;
				}
				if (k.supers_used !== undefined) {
					out += `Supers remaining: ${ss.max_supers - k.supers_used}<br>`;
				}
				if (k.power_bombs_used !== undefined) {
					out += `PBs remaining: ${ss.max_power_bombs - k.power_bombs_used}<br>`;
				}
				if (out != "") {
					item_info.appendChild(createDiv(`<small>${out}</small>`));
				}
				if (k.relevant_flags) {
					let flagContainer = document.createElement("small");
					let flagSpan0 = document.createElement("span");
					flagSpan0.innerText = "Relevant flags: ";
					flagContainer.appendChild(flagSpan0);
					for (let f in k.relevant_flags) {
						if (f != 0) {
							let flagSeparator = document.createElement("span");
							flagSeparator.innerText = ", ";
							flagContainer.appendChild(flagSeparator);
						}
						let flagA = document.createElement("a");
						let flagSpan = document.createElement("span");
						flagSpan.innerText = k.relevant_flags[f];
						flagA.appendChild(flagSpan);
						flagA.href = "#";
						flagA.onclick = function () {
							showFlag(c.details, k.relevant_flags[f]);
						};
						flagContainer.appendChild(flagA);
					}
					flagContainer.appendChild(document.createElement("br"));
					item_info.appendChild(flagContainer);
				}
			}
			item_info.appendChild(createHtmlElement(`<div class="category">RETURN ROUTE</div>`));
			lastRoom = null;
			lastNode = null;
			for (let k of j.return_route) {
				let out = "";
				if (k.energy_used !== undefined) {
					out += `Energy still needed: ${k.energy_used + 1}<br>`;
				}
				if (k.reserves_used !== undefined) {
					out += `Reserves still needed: ${k.reserves_used}<br>`;
				}
				if (k.missiles_used !== undefined) {
					out += `Missiles still needed: ${k.missiles_used}<br>`;
				}
				if (k.supers_used !== undefined) {
					out += `Supers still needed: ${k.supers_used}<br>`;
				}
				if (k.power_bombs_used !== undefined) {
					out += `PBs still needed: ${k.power_bombs_used}<br>`;
				}
				if (out != "") {
					item_info.appendChild(createHtmlElement(`<small>${out}</small>`));
				}

				let strat_url = `/logic/room/${k.room_id}/${k.from_node_id}/${k.to_node_id}/${k.strat_id}`;
				let nodeStr;
				if (k.strat_id !== null) {
					nodeStr = `<a style="text-decoration:none" href="${strat_url}">${k.room}: ${k.node}</a><br>`;
				} else {
					nodeStr = `${k.room}: ${k.node}<br>`;
				}
				if (k.room != lastRoom || k.node != lastNode || k.strat_id !== null) {
					item_info.appendChild(createDiv(nodeStr));
					lastRoom = k.room;
					lastNode = k.node;
				}
				out = "";
				if (!k.strat_name.startsWith("Base ") && k.strat_name != "Base" && k.strat_name != "Leave Normally") {
					if (k.strat_notes) {
						let title = "";
						for (let i of k.strat_notes) {
							title += `${i} `;
						}
						out += `Strat: <a href=${strat_url}><abbr title="${title}">${k.strat_name}</abbr></a><br>`;
					} else {
						out += `Strat: <a href=${strat_url}>${k.strat_name}</a><br>`;
					}
				}
				if (out != "") {
					item_info.appendChild(createHtmlElement(`<small>${out}</small>`));
				}
				if (k.relevant_flags) {
					let flagContainer = document.createElement("small");
					let flagSpan0 = document.createElement("span");
					flagSpan0.innerText = "Relevant flags: ";
					flagContainer.appendChild(flagSpan0);
					for (let f in k.relevant_flags) {
						if (f != 0) {
							let flagSeparator = document.createElement("span");
							flagSeparator.innerText = ", ";
							flagContainer.appendChild(flagSeparator);
						}
						let flagA = document.createElement("a");
						let flagSpan = document.createElement("span");
						flagSpan.innerText = k.relevant_flags[f];
						flagA.appendChild(flagSpan);
						flagA.href = "#";
						flagA.onclick = function () {
							showFlag(c.details, k.relevant_flags[f]);
						};
						flagContainer.appendChild(flagA);
					}
					flagContainer.appendChild(document.createElement("br"));
					item_info.appendChild(flagContainer);
				}
			}
		}
		si.appendChild(item_info);
	}
	function showFlag(details, flagName) {
		for (let stepNum in details) {
			let stepData = details[stepNum];
			for (let flagData of stepData.flags) {
				if (flagData.flag == flagName) {
					show_item_details(flagName, flagData.location, stepNum, flagData);
				}
			}
		}
	}
	items: for (let v of c.all_items) {
		if (v.item == "Nothing") { continue; }
		let os = lookupOffset(v.location.room, v.location.node);
		if (os) {
			v.location.coords[0] += os[0];
			v.location.coords[1] += os[1];
		}
		let el = document.createElement("div");
		el.className = "icon";
		el.style.left = v.location.coords[0] * 24 + 24 + 4 + "px";
		el.style.top = v.location.coords[1] * 24 + 24 + 4 + "px";
		el.style.backgroundPositionX = `-${item_plm[v.item] * 16}px`;
		let i = null;
		let j = null;
		for (let l in c.details) {
			for (let k of c.details[l].items) {
				if (k.location.room_id == v.location.room_id && k.location.node_id == v.location.node_id) {
					i = l;
					j = k;
					break;
				}
			}
		}
		el.onclick = ev => {
			show_item_details(v.item, v.location, i, j);
		};
		document.getElementById("overlay").appendChild(el);
		el = document.createElement("div");
		el.className = "popup";
		el.innerHTML = `<b>${v.item}</b><br><small>${v.location.room}</small><br>`;
		let fin = false;
		out:
		for (let i in c.details) {
			for (let j of c.details[i].items) {
				if (j.location.room_id == v.location.room_id && j.location.node_id == v.location.node_id) {
					el.innerHTML += `Step: ${c.details[i].step}<br>`;
					fin = true;
					break out;
				}
			}
		}
		if (!fin) {
			el.innerHTML += "Route unavailable<br>";
		}
		el.style.left = v.location.coords[0] * 24 + 56 + "px";
		el.style.top = v.location.coords[1] * 24 + 8 + "px";
		document.getElementById("overlay").appendChild(el);
	}
});
