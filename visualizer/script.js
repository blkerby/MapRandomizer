function lookupOffset(room, node) {
	key = room + ": " + node
	return offsets[key];
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
let icon = id => {
	let el = document.createElement("span");
	el.className = "ui-icon";
	el.style.backgroundPositionX = `-${id * 16}px`;
	return el;
}

screen.orientation.onchange = ev => {
	const h = screen.availHeight;
	if (h < 600+32)
		document.getElementById("sidebar-info").style.maxHeight = h-32 + "px";
	else
		document.getElementById("sidebar-info").style.maxHeight = "600px";
}

document.getElementById("ship").onchange = ev => {
	document.getElementById("gunship").style.visibility = ev.target.checked ? "visible" : "hidden";
}
document.getElementById("start").onchange = ev => {
	document.getElementById("helm").style.visibility = ev.target.checked ? "visible" : "hidden";
}

let startitems = 0;
let startbase = {x:0,y:0};
function moveStart() {
	if (startitems == 0)
		return;

	var l = 0;
	var e = document.getElementById("f_DefeatedBombTorizo");
	var itemson = document.getElementById("items").checked;
	if (startitems>2) {
		if (e.style.visibility == "visible")
			l++;
		if (itemson)
			l++;
	} else {
		if (itemson)
			l = startitems;
	}
	e = document.getElementById("helm");
	let ox = 0;
	let oy = 0;
	if (l==1) {
		ox -=12;
	} else if (l==2) {
		oy += 6;
	}
	e.style.left = startbase.x+ox+ "px";
	e.style.top = startbase.y+oy+ "px";
}
function toggleitemvis(ev) {
	togglevis(ev);
	moveStart();
	
	var e = document.getElementById("f_DefeatedBombTorizo");
	var x= Number(e.style.left.substring(0, e.style.left.length-2));
	if (ev.target.checked) {
		e.style.left = x + 6 + "px";
	} else {
		e.style.left = x - 6 + "px";
	}
}
function toggleflagvis(ev) {
	var e = document.getElementById("f_DefeatedBombTorizo");
	var eVis = e.style.visibility;
	togglevis(ev);
	moveStart();
	if (e.style.visibility != eVis) {
		var i = document.getElementById("Bomb Torizo Room: Item");
		if (i) {
			var x= Number(i.style.left.substring(0, i.style.left.length-2));
			if (e.style.visibility == "visible") {
				i.style.left = x - 6 + "px";
			} else {
				i.style.left = x + 6 + "px";
			}
		}
	}
}
function toggleobjectives(ev) {
	var on = ev.target.checked;
	for (var e of document.getElementsByClassName("objectives"))
		e.src = on ? e.classList[1] + "obj.png" : e.classList[1] + ".png";
}
function togglevis(ev) {
	var toggles = document.getElementsByClassName(ev.target.id);
	for (var e of toggles) {
		e.style.visibility = ev.target.checked ? "visible" : "hidden";
	}
}
document.getElementById("settingsCog").onclick = ev => {
	let f = document.getElementById("settingsForm")
	f.style.display = f.style.display == "none" ? "block" : "none";
}
loadForm(document.getElementById("settingsForm"));
loadForm(document.getElementById("helpForm"));
if (!document.getElementById("showonce").checked)
	document.getElementById("msg-wrap").style.display = "flex";

fetch(`../spoiler.json`).then(c => c.json()).then(c => {
	flagtypes["objectives"] = c.objectives;
	flagtypes["objectives"].push("f_DefeatedMotherBrain");
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
	if (c.summary.length == 0)
		gen_obscurity(null);
	else
		gen_obscurity(1);
	
	let show_item_details = (item_name, loc, i, j) => {
		if (j !== null) {
			document.getElementById("path-overlay").innerHTML = ""
			showRoute(j.return_route, "yellow");
			showRoute(j.obtain_route);
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
			
			let item_list = document.createElement("div");
			item_list.className = "item-list";
			let s = [ss.max_missiles, ss.max_supers, ss.max_power_bombs, Math.floor(ss.max_energy / 100), ss.max_reserves / 100];
			let ic = [1, 2, 3, 0, 20];
			for (let i in s) {
				if (s[i] > 0) {
					item_list.appendChild(icon(ic[i]));
					let count = document.createElement("span");
					count.innerHTML = s[i] + " ";
					item_list.appendChild(count);
				}
			}
			for (let i of ss.items) {
				if (i == "Nothing") { continue; }
				if (!ic.includes(item_plm[i])) {
					item_list.appendChild(icon(item_plm[i]));
				}
			}
			si.appendChild(item_list);
			
			flagIcons(si, ss.flags);

			let collectible_header = document.createElement("div");
			collectible_header.className = "category";
			collectible_header.innerHTML = "COLLECTIBLE ON THIS STEP";
			si.appendChild(collectible_header);


			item_list = document.createElement("div");
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
		
		flagIcons(si, c.summary[i].flags, j);

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
			routeData(item_info, j.obtain_route, ss);
				
			item_info.appendChild(createHtmlElement(`<div class="category">RETURN ROUTE</div>`));
			routeData(item_info, j.return_route);
		}
		si.appendChild(item_info);
	}
	function consumableData(k, ss=null) {
		let remstr = ss == null ? "still needed" : "remaining";
		let out = "";
		if (k.energy_used !== undefined) {
			if (ss == null)
				out += `Energy ${remstr}: ${k.energy_used + 1}<br>`;
			else
				out += `Energy ${remstr}: ${ss.max_energy - k.energy_used}<br>`;
		}
		if (k.reserves_used !== undefined) {
			if (ss == null)
				out += `Reserves ${remstr}: ${k.reserves_used}<br>`;
			else
				out += `Reserves ${remstr}: ${ss.max_reserves - k.reserves_used}<br>`;
		}
		if (k.missiles_used !== undefined) {
			if (ss == null)
				out += `Missiles ${remstr}: ${k.missiles_used}<br>`;
			else
				out += `Missiles ${remstr}: ${ss.max_missiles - k.missiles_used}<br>`;
		}
		if (k.supers_used !== undefined) {
			if (ss == null)
				out += `Supers ${remstr}: ${k.supers_used}<br>`;
			else
				out += `Supers ${remstr}: ${ss.max_supers - k.supers_used}<br>`;
		}
		if (k.power_bombs_used !== undefined) {
			if (ss == null)
				out += `PBs ${remstr}: ${k.power_bombs_used}<br>`;
			else
				out += `PBs ${remstr}: ${ss.max_power_bombs - k.power_bombs_used}<br>`;
		}
		return out;
	}
	function routeData(p, route, ss=null) {
		let lastRoom=null, lastNode=null;
		for (let k of route) {
			let strat_url = `/logic/room/${k.room_id}/${k.from_node_id}/${k.to_node_id}/${k.strat_id}`;
			let nodeStr;
			let out = "";

			if (ss == null)	{
				out = consumableData(k);
				if (out != "") {
					p.appendChild(createHtmlElement(`<small>${out}</small>`));
				}
			}

			if (k.strat_id !== null) {
				nodeStr = `<a style="text-decoration:none" href="${strat_url}">${k.room}: ${k.node}</a><br>`;
			} else {
				nodeStr = `${k.room}: ${k.node}<br>`;
			}
			if (k.room != lastRoom || k.node != lastNode || k.strat_id !== null) {
				p.appendChild(createDiv(nodeStr));
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
					if (k.strat_id !== null) {
						out += `Strat: <a href=${strat_url}><abbr title="${title}">${k.strat_name}</abbr></a><br>`;
					} else {
						out += `Strat: <abbr title="${title}">${k.strat_name}</abbr><br>`;
					}
				} else {
					if (k.strat_id !== null) {
						out += `Strat: <a href=${strat_url}>${k.strat_name}</a><br>`;
					} else {
						out += `Strat: ${k.strat_name}<br>`;
					}
				}
			}
			if (out != "") {
				p.appendChild(createHtmlElement(`<small>${out}</small>`));
			}
			if (ss != null)	{
				out = consumableData(k, ss);
				if (out != "") {
					p.appendChild(createHtmlElement(`<small>${out}</small>`));
				}
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
				p.appendChild(flagContainer);
			}
		}
	}
	function flagIcons(p, flags, j=null) {
		for (i in flags) {
			let x = flags[i]
			let f = x;
			if (x.flag != null)
				f = x.flag;

			if (f == "f_TourianOpen" || f == "f_MotherBrainGlassBroken" || f == "f_AllItemsSpawn" || f == "f_AcidChozoWithoutSpaceJump" || f.includes("f_KilledZebetites"))
				continue;

			let e = document.createElement("img");
			e.src = f + ".png";
			
			e.className = "ui-flag"
			if (j && j.flag && j.flag == f)
				e.classList.add("selected");
			e.onclick = ev => {
				showFlag(c.details, f);
			}
			p.appendChild(e);
		}
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
		if (flagName == "f_DefeatedMotherBrain") {
			showEscape();
		}
	}
	function showEscape() {
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
	function showRoute(to, color="white") {
		if (to == null || to.length == 0)
			return;

		path = "";
		for (let k of to) {
			if (k.coords) {
				let x = k.coords[0] * 24 + 24 + 12;
				let y = k.coords[1] * 24 + 24 + 12;
				path += `${path == "" ? "M" : "L"}${x} ${y} `;
			}
		}
		document.getElementById("path-overlay").innerHTML += `<path d="${path}" stroke="black" fill="none" stroke-linejoin="round" stroke-width="4"/>`
		document.getElementById("path-overlay").innerHTML += `<path d="${path}" stroke="${color}" fill="none" stroke-linejoin="round" stroke-width="2"/>`
	}
	function hideRoom() {
		var el = document.getElementById("room-info");
		el.classList.add("hidden")
		el.innerText = "";
	}
	function hubRoute() {
		document.getElementById("path-overlay").innerHTML = ""
		// escape mode
		if (c.summary.length ==0)
			return;

		gen_obscurity(1);

		if (c.hub_obtain_route.length == 0)
			return;
		
		showRoute(c.hub_return_route, "yellow");
		showRoute(c.hub_obtain_route);


		let si = document.getElementById("sidebar-info")
		si.scrollTop = 0;
		si.innerHTML = "";
		let title = document.createElement("div");
		title.className = "sidebar-title";
		title.innerHTML = `Hub route`;
		si.appendChild(title);
		

		let previous_header = document.createElement("div");
		previous_header.className = "category";
		previous_header.innerHTML = "PREVIOUSLY COLLECTED";
		si.appendChild(previous_header);

		let ss = c.details[0].start_state;
		
		let item_list = document.createElement("div");
		item_list.className = "item-list";
		let s = [ss.max_missiles, ss.max_supers, ss.max_power_bombs, Math.floor(ss.max_energy / 100), ss.max_reserves / 100];
		let ic = [1, 2, 3, 0, 20];
		for (let i in s) {
			if (s[i] > 0) {
				item_list.appendChild(icon(ic[i]));
				let count = document.createElement("span");
				count.innerHTML = s[i] + " ";
				item_list.appendChild(count);
			}
		}
		for (let i of ss.items) {
			if (i == "Nothing") { continue; }
			if (!ic.includes(item_plm[i])) {
				item_list.appendChild(icon(item_plm[i]));
			}
		}
		si.appendChild(item_list);
		
		flagIcons(si, ss.flags);
		let item_info = document.createElement("div");
		item_info.appendChild(createHtmlElement(`<div class="category">OBTAIN ROUTE</div>`));
		routeData(item_info, c.hub_obtain_route);
				
		item_info.appendChild(createHtmlElement(`<div class="category">RETURN ROUTE</div>`));
		routeData(item_info, c.hub_return_route);
		si.appendChild(item_info);
	}


	let helmx = 0, helmy = 0;
	starticon: {
		let sr = null, e = null, ri = c.start_location.room_id, ni = c.start_location.node_id, i=-1, x=0, y=0;
		let n = c.start_location.name;
		for (i in c.all_rooms) {
			if (ri ==c.all_rooms[i].room_id )
			{
				// only used when start location == hub
				x = c.all_rooms[i].coords[0]*24 + 24 + Math.floor(c.start_location.x / 16) * 24;
				y = c.all_rooms[i].coords[1]*24 + 24 + Math.floor(c.start_location.y / 16) * 24;
				break;
			}
		}
		if (n == "Ship") {
			i = 1;
			x = c.all_rooms[i].coords[0]*24+24;
			y = c.all_rooms[i].coords[1]*24+24;
			x += 96;
			y += 72;	
		} else if (n == "Bomb Torizo Room") {
			startitems=3;
		} else if (n == "") {
			n = "Mother Brain Room";
			i = 248;
			x = c.all_rooms[i].coords[0]*24+24;
			y = c.all_rooms[i].coords[1]*24+24;
		}

		sr = c.all_rooms[i];
		startbase.x = x;
		startbase.y = y;
		for (i in c.all_items) {
			let loc = c.all_items[i].location;
			if (loc.room_id == ri) {
				let os = lookupOffset(loc.room, loc.node);
				let lx = loc.coords[0]*24+24+Math.round(os[0])*24;
				let ly = loc.coords[1]*24+24+Math.round(os[1])*24;
				if (lx == x && ly== y) {
					if (startitems == 0) {
						startitems++;
					} else if (startitems ==1) {
						startitems++;
					}
				}
			}
		}

		e = document.createElement("img");
		e.src = "helm.png";
		e.id = "helm"
		e.className = "start";
		e.style.left =  x + "px";
		e.style.top =  y + "px";
		helmx = x;
		helmy = y;
		e.style.visibility = document.getElementById("start").checked ? "visible" : "hidden";
		e.onclick = ev => {
			hubRoute();
		}
		e.onpointermove = ev => {
			hideRoom();
		}
		document.getElementById("overlay").appendChild(e);
		e = document.createElement("div");
		e.className = "popup";
		e.innerHTML = `<b>Starting Location</b><br><small>${sr.room}</small><br>`;
		e.style.left = x+24+ "px";
		e.style.top = y+ "px";
		document.getElementById("overlay").appendChild(e);
	}
		
	shipicon: {
		sr = c.all_rooms[1];
		e = document.createElement("img");
		e.src = "gunship.png";
		e.id = "gunship"
		e.className = "ship";
		
		x = sr.coords[0]*24+116;
		y = sr.coords[1]*24+124;
		e.style.left = x+"px";
		e.style.top = y+"px";
		e.style.visibility = document.getElementById("ship").checked ? "visible" : "hidden";
		e.onclick = ev => {
			document.getElementById("path-overlay").innerHTML = ""
			show_overview();
			showEscape();
			gen_obscurity(null);
		}
		e.onpointermove = ev => {
			hideRoom();
		}
		document.getElementById("overlay").appendChild(e);
		e = document.createElement("div");
		e.className = "popup";
		e.innerHTML = `<b>Ship</b><br><small>${sr.room}</small><br>`;
		e.style.left = x + 48 +"px";
		e.style.top = y + "px";
		document.getElementById("overlay").appendChild(e);
	}

	flags: {
		for (i in roomFlags) {
			e = document.createElement("img");
			let rf = roomFlags[i];
			let f = rf[0];
			let obj = false;
			if (f == "f_ZebesAwake")
				continue;

			for (j in c.all_rooms)
			{
				if (i == c.all_rooms[j].room)
					break;
			}
			sr = c.all_rooms[j];
			e.className = "flag";
			e.id = f;
			let fc = null;
			for (ic in flagtypes) {
				for (x of flagtypes[ic]) {
					if (x == f) {
						e.classList.add(ic);
						
						if (ic == "objectives")
							obj = true;
						else {
							fc = ic;
							e.style.visibility =  document.getElementById(ic).checked ? "visible" : "hidden";
						}
						break;
					}
				}
			}
			if (obj && document.getElementById("objectives").checked)
				e.src = fc + "obj.png"
			else
				e.src = fc + ".png"
			
			ox = 0;
			if (f == "f_DefeatedBombTorizo"  && document.getElementById("items").checked)
				ox += 6;
			e.style.left = (sr.coords[0]+rf[2])*24+24+ox+"px";
			e.style.top = (sr.coords[1]+rf[3])*24+24+"px";
			

			e.onclick = ev => {
				showFlag(c.details, f);
			}
			e.onpointermove = ev => {
				hideRoom();
			}
			document.getElementById("overlay").appendChild(e);

			e = document.createElement("div");
			e.className = "popup";
			e.innerHTML = `<b>${rf[1]}</b><br><small>${sr.room}</small><br>`;
			if (obj == true)
				e.innerHTML += "Objective<br>";
			let fin = false;
			out:
			for (i in c.summary) {
				for (j in c.summary[i].flags) {
					if (c.summary[i].flags[j]['flag'] == f) {
						e.innerHTML += `Step: ${c.summary[i].step}<br>`;
						fin = true;
						break out;
					}
				}
			}
			if (!fin) {
				e.innerHTML += "Route unavailable<br>";
			}
			e.style.left = (sr.coords[0]+rf[2]) * 24+48 + "px";
			e.style.top = (sr.coords[1]+rf[3])*+24+24 + "px";
			document.getElementById("overlay").appendChild(e);
		}
	}

	items: {
		for (let v of c.all_items) {
			if (v.item == "Nothing") { continue; }
			let os = lookupOffset(v.location.room, v.location.node);
			if (os) {
				v.location.coords[0] += os[0];
				v.location.coords[1] += os[1];
			}
			e = document.createElement("div");
			e.className = "icon";
			e.classList.add("items");
			e.classList.add(v.item);
			e.id = v.location.room+": "+v.location.node;

			let checked = document.getElementById("items").checked;
			e.style.visibility =  checked ? "visible" : "hidden";

			let ox = 0;
			if (e.id == "Bomb Torizo Room: Item" && document.getElementById("f_DefeatedBombTorizo").style.visibility == "visible") 
				ox -=6;
			e.style.left = v.location.coords[0] * 24 + 24 + 4 + ox + "px";
			e.style.top = v.location.coords[1] * 24 + 24 + 4 + "px";
			

			e.style.backgroundPositionX = `-${item_plm[v.item] * 16}px`;
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
			e.onclick = ev => {
				show_item_details(v.item, v.location, i, j);
			};
			e.onpointermove = ev => {
				hideRoom();
			}
			document.getElementById("overlay").appendChild(e);
			e = document.createElement("div");
			e.className = "popup";
			e.innerHTML = `<b>${v.item}</b><br><small>${v.location.room}</small><br>`;
			let fin = false;
			out:
			for (let i in c.details) {
				for (let j of c.details[i].items) {
					if (j.location.room_id == v.location.room_id && j.location.node_id == v.location.node_id) {
						e.innerHTML += `Step: ${c.details[i].step}<br>`;
						fin = true;
						break out;
					}
				}
			}
			if (!fin) {
				e.innerHTML += "Route unavailable<br>";
			}
			e.style.left = v.location.coords[0] * 24 + 56 + "px";
			e.style.top = v.location.coords[1] * 24 + 8 + "px";
			document.getElementById("overlay").appendChild(e);
			if (screen.availHeight < 600+32)
			document.getElementById("sidebar-info").style.maxHeight = screen.availHeight-32 + "px";
		}
	}
	moveStart();

	// input
	let page_x = -helmx+document.documentElement.clientWidth/2;
	let page_y = -helmy+document.documentElement.clientHeight/2;
	let el = document.getElementById("room-info");
	let dragged = false;
	let scale = 1, dm = 0;
	let m = document.getElementById("map");
	let evCache = [];
	let odist = -1;

	transfo();
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
				if (el.innerText.length != "" && c.hub_location_name.includes(el.innerText)) {
					el.innerHTML += " (Hub)";
				}
				return;
			}
		}
		el.classList.add("hidden")
		el.innerText = "";
	}
	function click() {
		if (!dragged) {
			// deselect
			show_overview();
			document.getElementById("path-overlay").innerHTML = ""
		}
	}
	function dblclick() {
		if (!el.classList.contains("hidden")) {
			window.open("/logic/room/" + el.dataset.roomId);
		}
	}
	function up(ev) {
		if (dragged)
			el.classList.add("hidden");
		
		evCache.splice(evCache.findIndex((cached) => cached.pointerID == ev.pointerID), 1)
		dragged = false;
	}
	m.onpointerdown = ev => {
		if (ev.button != 0)
			return;

		ev.preventDefault();
		evCache.push(ev);
		dragged = false;
		dm = 0;
		if (evCache.length == 2) {
			let dx = Math.abs(evCache[0].x-evCache[1].x);
			let dy = Math.abs(evCache[0].y-evCache[1].y);
			odist = Math.sqrt(dx**2+dy**2);
		}
	}
	let fclick = true, timer = null, oldroom = null;
	m.onpointerup = ev => {
		if (ev.button != 0)
			return;
		else
			ev.preventDefault();
		
		if (evCache.length == 1) {
			
			hover(ev);
			click();
			if (fclick) {
				timer = setTimeout(function (){
					fclick = true;
				}, 500);
				fclick = false;
				oldroom = el.innerText;
			} else {	
				fclick = true;
				if (timer)
					clearTimeout(timer);
				if (oldroom == el.innerText)
					dblclick();
				oldroom = null;
			}
		}
		up(ev);
	}
	document.body.onpointerleave = ev => {
		up(ev);
	}
	document.body.onpointerup = ev => {
		if (ev.button != 0)
			return;
		else
			ev.preventDefault();
		up(ev);
	}
	m.onpointermove = ev => {
		ev.preventDefault();
		if (evCache.length == 2) {
			if (ev.button == 0)
				return;
			var dx = Math.abs(evCache[0].x - evCache[1].x);
			var dy = Math.abs(evCache[0].y - evCache[1].y);
			var dist = Math.sqrt(dx**2 + dy**2);
			var delta = odist-dist;
			let i = evCache.findIndex((e) => e.pointerId == ev.pointerId);
			evCache[i] = ev;
			zm((evCache[0].x+evCache[1].x)/2, (evCache[0].y+evCache[1].y)/2,delta*2);
			odist = dist;
		} else if (evCache.length == 1) {
			dm += Math.abs(ev.x - evCache[0].x);
			dm += Math.abs(ev.y - evCache[0].y);
			if (dm > 3)
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
	document.getElementById("helpForm").onclick = ev => {
		ev.stopPropagation();
	}
	if (screen.availHeight < 600+32)
		document.getElementById("sidebar-info").style.maxHeight = screen.availHeight-32 + "px";
});
