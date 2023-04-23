const offsets = {
	"Power Bomb (blue Brinstar) (unlocked)": [2,2],
	"Morphing Ball": [4,2],
	"Missile (lava room)": [2,1],
	"Missile (Draygon)": [1,0],
	"Missile (green Maridia tatori)": [2,1],
	"Energy Tank, Mama turtle": [1,0],
	"Missile (green Brinstar behind reserve tank)": [0.75,0],
	"Missile (green Brinstar behind missile)": [1.25,0],
	"Reserve Tank, Brinstar": [0,0],
	"Energy Tank, Firefleas": [2,5],
	"Super Missile (green Maridia)": [1,2],
	"Missile (green Maridia shinespark)": [0,3],
	"Energy Tank, Gauntlet": [5,0],
	"Missile (Wrecked Ship middle)": [0,5],
	"Missile (yellow Maridia false wall)": [2,2],
	"Missile (Wrecked Ship top) (unlocked)": [2,0],
	"Missile (Kraid)": [2,0],
	"Missile (blue Brinstar top)": [0.25,0],
	"Missile (blue Brinstar behind missile)": [-0.25,0],
	"Missile (Norfair Reserve Tank)": [0.5,0],
	"Missile (green Brinstar pipe)": [3,1],
	"Energy Tank, Crocomire": [7,0],
	"Energy Tank, Brinstar Ceiling": [1,2],
	"Missile (blue Brinstar middle)": [2,2],
	"Missile (lower Norfair near Wave Beam)": [0,2],
	"Power Bomb (red Brinstar spike room)": [1,0],
	"Missile (red Brinstar spike room)": [0,0],
	"Super Missile (pink Maridia)": [5,4],
	"Missile (pink Maridia)": [4,4],
	"Screw Attack": [0,2],
	"Missile (bubble Norfair)": [1,3],
	"Missile (yellow Maridia super missile)": [-0.25,2],
	"Super Missile (yellow Maridia)": [0.25,2],
	"Missile (Speed Booster)": [11,1],
	"Reserve Tank, Maridia": [0.25,0],
	"Missile (left Maridia sand pit room)": [-0.25,0],
	"Missile (right Maridia sand pit room)": [0,0],
	"Power Bomb (right Maridia sand pit room)": [1,1],
	"Missile (below Crocomire)": [3,0],
	"Missile (Wave Beam)": [1,0],
	"Power Bomb (red Brinstar sidehopper room)": [0,1],
	"Missile (outside Wrecked Ship bottom)": [0,5],
	"Missile (outside Wrecked Ship middle)": [0,2],
	"Missile (outside Wrecked Ship top)": [1,0],
	"Missile (green Brinstar below super missile)": [1,1],
	"Super Missile (green Brinstar top)": [0,0],
	"Plasma Beam": [1,2],
	"Missile (Crateria gauntlet left)": [-0.25,1],
	"Missile (Crateria gauntlet right)": [0.25,1],
	"Spring Ball": [1,1],
	"Power Bomb (pink Brinstar)": [0,1],
	"Missile (Crateria bottom) (unlocked)": [0,1],
	"Missile (Mickey Mouse Room)": [2,1],
	"Grapple Beam": [0,2],
	"Super Missile (Golden Torizo)": [1,0],
	"Missile (Golden Torizo)": [0,0],
	"Right Super, Wrecked Ship (unlocked)": [3,0],
	"Energy Tank, Botwoon": [3,0],
	"Energy Tank, Terminator": [0,2],
	"Missile (Grapple Beam)": [4,0],
	"Missile (bubble Norfair green door)": [1,0],
	"Missile (lower Norfair above fire flea room)": [2,0],
	"Power Bomb (Crateria surface)": [1,0],
	"Power Bomb (green Brinstar bottom)": [3,7],
	"Super Missile (Crateria)": [3,0],
	"Super Missile (pink Brinstar)": [1,8],
	"Energy Tank (Hi-Jump Boots)": [1,0],
	"Missile (Hi-Jump Boots)": [0,0],
	"Missile (pink Brinstar top)": [2,3],
	"Missile (pink Brinstar bottom)": [2,6],
	"Charge Beam": [2,7],
	"Missile (Gravity Suit) (unlocked)": [3,2],
	"Reserve Tank, Wrecked Ship (unlocked)": [5,0],
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
	"ReserveTank": 20
};
let change_item = (old_plm,id) => {
	let old_id = ((old_plm - 0xEED7) / 4) % 21;
	return old_plm + (id - old_id) * 4;
}
let item_addrs;
fetch(`item_addrs.json`).then(c => c.json()).then(c => {
	item_addrs = c;
})
let plando_changes = {};
let doors;
fetch(`doors.json`).then(c => c.json()).then(c => {
	doors = c;
}).then(_ => fetch(`../spoiler.json`)).then(c => c.json()).then(c => {
	// generate map
	let map = new Array(72*72);
	for (let i in c.all_rooms) {
		let v = c.all_rooms[i];
		for (let y = 0; y < v.map.length; y++) {
			for (let x = 0; x < v.map[y].length; x++) {
				if (v.map[y][x] != 0) {
					map[(v.coords[1]+y)*72+(v.coords[0]+x)] = +i;
				}
			}
		}
	}
	let step_limit = null;
	window.gen_obscurity = (sl) => {
		step_limit = sl;
		// generate obscurity overlay
		let ov = document.getElementById("obscure-overlay");
		let ctx = ov.getContext("2d");
		let img = ctx.createImageData(72,72);
		if (step_limit === null) {
			ctx.putImageData(img, 0, 0);
			return;
		}
		for (let i = 0; i < 72 * 72; i++) {
			img.data[i*4+3] = 0xFF; // opaque
		}
		for (let v of c.all_rooms) {
			for (let y = 0; y < v.map.length; y++) {
				for (let x = 0; x < v.map[y].length; x++) {
					if (v.map[y][x] == 1) {
						let addr = (v.coords[1]+y)*72+(v.coords[0]+x);
						if (v.map_bireachable_step[y][x] < step_limit) {
							img.data[addr*4+3] = 0x00; // transparent
						} else if (v.map_reachable_step[y][x] < step_limit) {
							img.data[addr*4+3] = 0x7F; // semiopaque
						} else {
							img.data[addr*4+3] = 0xFF; // opaque
						}
					}
				}
			}
		}
		ctx.putImageData(img, 0, 0);
		// TODO: hide items and unfuck overlay
	}
	gen_obscurity(null);
	let el = document.getElementById("room-info");
	let dragging = false;
	document.getElementById("map").ondragstart = ev => {
		return false;
	}
	document.body.onmousedown = ev => {
		dragging = true;
	}
	document.body.onmouseup = ev => {
		dragging = false;
	}
	document.body.onmouseleave = ev => {
		dragging = false;
		el.classList.add("hidden")
	}
	let page_x = 0;
	let page_y = 0;
	document.body.onmousemove = ev => {
		if (dragging) {
			page_x += ev.movementX
			page_y += ev.movementY;
			document.body.style.setProperty("--tx", page_x + "px");
			document.body.style.setProperty("--ty", page_y + "px");
		} else {
			let x = ((ev.offsetX / 24)|0) - 1;
			let y = ((ev.offsetY / 24)|0) - 1;
			if (x >= 0 && x < 72 && y >= 0 && y < 72) {
				let tile = map[y*72+x];
				if (tile > 0) {
					el.innerText = c.all_rooms[tile].room;
					el.style.left = ev.offsetX + 16 + "px";
					el.style.top = ev.offsetY + "px";
					el.classList.remove("hidden");
					return;
				}
			}
			el.classList.add("hidden")
		}
	}
	window.generate_rom = async () => {
		const rom_data = await localforage.getItem("baseRomData") ?? await localforage.getItem("vanillaRomData");
		if (!rom_data) {
			alert("Please set the ROM on the previous page.");
			return;
		}
		const form = new FormData();
		form.append("room_palettes", "vanilla");
		const rom = new Blob([rom_data], { type: "" });
		form.append("rom", rom);
		let c = await fetch("../../customize", { body: form, method: "post" });
		let blob = await c.blob();
		// change up the item locations
		let ab = await blob.arrayBuffer();
		let bytes = new Uint8Array(ab);
		for (let i in plando_changes) {
			let offset = +item_addrs[i];
			console.log(offset, item_addrs[i]);
			let value = item_plm[plando_changes[i]];
			let orig = bytes[offset] | (bytes[offset+1]<<8)
			let n = change_item(orig, value);
			console.log(orig, value, n);
			bytes[offset] = n;
			bytes[offset+1] = n>>8;
			console.log(bytes[offset] | (bytes[offset+1]<<8));
		}
		blob = new Blob([ab]);
		let a = document.createElement("a");
		a.href = URL.createObjectURL(blob);
		a.setAttribute("download", "plando.sfc");
		a.click();
	}
	let icon = id => `<div class="ui-icon" style="background-position-x: -${id*16}px"></div>`;
	let print_route = () => {
		let si = document.getElementById("sidebar-info");
		let out = "";
		let seen = new Set();
		for (let i of c.summary) {
			out += `<div onclick="gen_obscurity(${i.step})">`;
			out += `<div class="category">STEP ${i.step}</div>`;
			for (let j of i.items) {
				if (!seen.has(j.item)) {
					out += icon(item_plm[j.item]);
					seen.add(j.item);
				}
			}
			out += `</div>`;
		}
		out += `<div onclick="gen_obscurity(null)"><div class="category">OTHER ITEMS</div></div>`;
		out += `<button onclick="generate_rom()">Download ROM</button>`;
		si.innerHTML = out;
	}
	print_route();
	document.getElementById("map").onclick = ev => {
		if (el.innerText == "Mother Brain Room") {
			let path = "";
			for (let i of c.escape.animals_route ?? []) {
				for (let k of [i.from, i.to]) {
					let r = c.all_rooms.find(c => c.room == k.room);
					let xl = r.coords[0];
					let yl = r.coords[1];
					let o = doors.find(c => c.name == k.node);
					if (offsets[k.node]) {
						xl += offsets[k.node][0];
						yl += offsets[k.node][1];
					} else if (o && o.nodeAddress) {
						if (o.x !== undefined && o.y !== undefined) {
							xl += o.x; yl += o.y;
						}
					} else { continue; }
					let x = xl * 24 + 24 + 12;
					let y = yl * 24 + 24 + 12;
					path += `${path == "" ? "M" : "L"}${x} ${y} `;
				}
			}
			for (let i of c.escape.ship_route) {
				for (let k of [i.from, i.to]) {
					let r = c.all_rooms.find(c => c.room == k.room);
					let xl = r.coords[0];
					let yl = r.coords[1];
					let o = doors.find(c => c.name == k.node);
					if (offsets[k.node]) {
						xl += offsets[k.node][0];
						yl += offsets[k.node][1];
					} else if (o && o.nodeAddress) {
						if (o.x !== undefined && o.y !== undefined) {
							xl += o.x; yl += o.y;
						}
					} else { continue; }
					let x = xl * 24 + 24 + 12;
					let y = yl * 24 + 24 + 12;
					path += `${path == "" ? "M" : "L"}${x} ${y} `;
				}
			}
			document.getElementById("path-overlay").innerHTML = `<path d="${path}" stroke="black" fill="none" stroke-linejoin="round" stroke-width="4"/>`
			document.getElementById("path-overlay").innerHTML += `<path d="${path}" stroke="cyan" fill="none" stroke-linejoin="round" stroke-width="2"/>`
		} else {
			// deselect
			print_route();
		}
	}
	document.getElementById("map").ondblclick = ev => {
		if (!el.classList.contains("hidden")) {
			window.open("https://wiki.supermetroid.run/" + encodeURIComponent(el.innerText), "_blank")
		}
	}
	items: for (let v of c.all_items) {
		if (v.location.node in offsets) {
			v.location.coords[0] += offsets[v.location.node][0];
			v.location.coords[1] += offsets[v.location.node][1];
		}
		let el = document.createElement("div");
		el.className = "icon";
		//console.log(v);
		el.style.left = v.location.coords[0] * 24 + 24 + 4 + "px";
		el.style.top = v.location.coords[1] * 24 + 24 + 4 + "px";
		el.style.backgroundPositionX = `-${item_plm[v.item] * 16}px`;
		let i = null;
		let j = null;
		for (let l in c.details) {
			for (let k of c.details[l].items) {
				if (k.location.node == v.location.node) {
					i = l;
					j = k;
					break;
				}
			}
		}
		let icon_el = el;
		el.onclick = ev => {
			if (j) {
				let path = "";
				for (let k of j.return_route) {
					let xl = k.coords[0];
					let yl = k.coords[1];
					let o = doors.find(c => c.name == k.node);
					if (offsets[k.node]) {
						xl += offsets[k.node][0];
						yl += offsets[k.node][1];
					} else if (o && o.nodeAddress) {
						if (o.x !== undefined && o.y !== undefined) {
							xl += o.x; yl += o.y;
						}
					} else { continue; }
					let x = xl * 24 + 24 + 12;
					let y = yl * 24 + 24 + 12;
					path += `${path == "" ? "M" : "L"}${x} ${y} `;
				}
				document.getElementById("path-overlay").innerHTML = `<path d="${path}" stroke="black" fill="none" stroke-linejoin="round" stroke-width="4"/>`
				document.getElementById("path-overlay").innerHTML += `<path d="${path}" stroke="yellow" fill="none" stroke-linejoin="round" stroke-width="2"/>`
				path = "";
				for (let k of j.obtain_route) {
					let xl = k.coords[0];
					let yl = k.coords[1];
					let o = doors.find(c => c.name == k.node);
					if (offsets[k.node]) {
						xl += offsets[k.node][0];
						yl += offsets[k.node][1];
					} else if (o && o.nodeAddress) {
						if (o.x !== undefined && o.y !== undefined) {
							xl += o.x; yl += o.y;
						}
					} else { continue; }
					let x = xl * 24 + 24 + 12;
					let y = yl * 24 + 24 + 12;
					path += `${path == "" ? "M" : "L"}${x} ${y} `;
				}
				document.getElementById("path-overlay").innerHTML += `<path d="${path}" stroke="black" fill="none" stroke-linejoin="round" stroke-width="4"/>`
				document.getElementById("path-overlay").innerHTML += `<path d="${path}" stroke="white" fill="none" stroke-linejoin="round" stroke-width="2"/>`
			}
			let si = document.getElementById("sidebar-info");
			si.innerHTML = `<div class="sidebar-title">${v.item}</div><div class="category">LOCATION</div>${v.location.room}<br><small>${v.location.node}</small>`;
			if (j) {
				si.innerHTML += `<div class="category">PREREQUISITES</div>`;
				let ss = c.details[i].start_state;
				let s = [ss.max_missiles, ss.max_supers, ss.max_power_bombs, ss.max_energy, ss.max_reserves];
				let ic = [1, 2, 3, 0, 20];
				for (let i in s) {
					if (s[i] > 0) {
						si.innerHTML += icon(ic[i]);
						si.innerHTML += s[i] + " ";
					}
				}
				for (let i of ss.items) {
					if (!ic.includes(item_plm[i])) {
						si.innerHTML += icon(item_plm[i]);
					}
				}
				si.innerHTML += `<div class="category">OBTAIN ROUTE</div>`;
				let f = k => {
					si.innerHTML += `${k.node}<br>`;
					let out = "";
					if (k.strat_name != "Base" && k.strat_name != "(Door transition)") {
						if (k.strat_notes) {
							let title = "";
							for (let i of k.strat_notes) {
								title += `${i} `;
							}
							out += `Strat: <abbr title="${title}">${k.strat_name}</abbr><br>`;
						} else {
							out += `Strat: ${k.strat_name}<br>`;
						}
					}
					if (k.energy_remaining) {
						out += `Energy remaining: ${k.energy_remaining}<br>`;
					}
					if (k.reserves_remaining) {
						out += `Reserves remaining: ${k.reserves_remaining}<br>`;
					}
					if (k.missiles_remaining) {
						out += `Missiles remaining: ${k.missiles_remaining}<br>`;
					}
					if (k.supers_remaining) {
						out += `Supers remaining: ${k.supers_remaining}<br>`;
					}
					if (k.power_bombs_remaining) {
						out += `PBs remaining: ${k.power_bombs_remaining}<br>`;
					}
					if (out != "") {
						si.innerHTML += `<small>${out}</small>`;
					}
				};
				for (let k of j.obtain_route) {
					f(k);
				}
				si.innerHTML += `<div class="category">RETURN ROUTE</div>`;
				for (let k of j.return_route) {
					f(k);
				}
			}
			si.innerHTML += `<div class="category">CHANGE ITEM</div>`;
			let out = "";
			out += `<select onchange="changeItem(this)">`;
			for (let i of Object.keys(item_plm)) {
				out += `<option ${i == v.item ? "selected" : ""}>${i}</option>`
			}
			out += `</select>`;
			window.changeItem = sel => {
				v.item = sel.value;
				icon_el.style.backgroundPositionX = `-${item_plm[sel.value] * 16}px`;
				plando_changes[v.location.node] = sel.value;
			}
			si.innerHTML += out;
		};
		document.getElementById("overlay").appendChild(el);
		el = document.createElement("div");
		el.className = "popup";
		el.innerHTML = `<b>${v.item}</b><br><small>${v.location.room}</small><br>`;
		let fin = false;
		out:
		for (let i in c.details) {
			for (let j of c.details[i].items) {
				if (j.location.node == v.location.node) {
					el.innerHTML += `Step: ${c.details[i].step}<br>`;
					let strats = new Set();
					let energy = null;
					let reserves = null;
					for (let k of j.obtain_route) {
						if (k.strat_name != "Base" && k.strat_name != "(Door transition)") {
							strats.add(k.strat_name);
						}
						if (k.energy_remaining) {
							energy = k.energy_remaining;
							reserves = k.reserves_remaining;
						}
					}
					if (energy) {
						let en = c.details[i].start_state.max_energy - energy;
						if (reserves) {
							en += c.details[i].start_state.max_reserves - reserves;
						}
						el.innerHTML += `Energy used: ${en}<br>`;
					}
					/*if (strats.size != 0) {
						el.innerHTML += `Strats: <br><small>${[...strats].join("<br>")}</small>`;
					}*/
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
