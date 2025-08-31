var spoiler = null;
var roomMap = new Map();
var nodeMap = new Map();

function lookupOffset(room_id, node_id) {
	key = room_id + ":" + node_id
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
document.getElementById("sidebar-info").onmousemove = ev => {
	let el = document.getElementById("room-info");
	el.classList.add("hidden");
	el.innerText = "";
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
		// TODO: adjust by only 6 but shift the item over the opposite way as well.
		ox -= 12;
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
	if (e === null)
		return;
	var x= Number(e.style.left.substring(0, e.style.left.length-2));
	if (ev.target.checked) {
		e.style.left = x + 6 + "px";
	} else {
		e.style.left = x - 6 + "px";
	}
}
function toggleflagvis(ev) {
	var e = document.getElementById("f_DefeatedBombTorizo");
	if (e === null)
		return;
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
function setDebugDataVisibility() {
	let checked = document.getElementById("debugDataCheckbox").checked;
	let debugData = document.getElementById("debugData");
	debugData.style.display = checked ? "block" : "none";
}

function changeDebugDataVertexId() {
	if (spoiler === null) {
		return;
	}
	let vertexId = parseInt(document.getElementById("debugVertexId").value);
	let vertexKey = spoiler.game_data.vertices[vertexId];
	if (vertexKey === undefined) {
		document.getElementById("debugRoomId").value = "";
		document.getElementById("debugNodeId").value = "";
		document.getElementById("debugObstacleMask").value = "";
		return;
	}
	document.getElementById("debugRoomId").value = vertexKey.room_id;
	document.getElementById("debugNodeId").value = vertexKey.node_id;
	document.getElementById("debugObstacleMask").value = vertexKey.obstacle_mask;
}

function changeDebugDataInput() {
	if (spoiler === null) {
		return;
	}
	let roomId = parseInt(document.getElementById("debugRoomId").value);
	let nodeId = parseInt(document.getElementById("debugNodeId").value);
	let obstacleMask = parseInt(document.getElementById("debugObstacleMask").value);
	// We could build and use a hash map for this, but it's not really necessary.
	document.getElementById("debugVertexId").value = "";
	for (vertexId in spoiler.game_data.vertices) {
		let key = spoiler.game_data.vertices[vertexId];
		if (key.room_id == roomId && key.node_id == nodeId 
			&& key.obstacle_mask == obstacleMask && key.actions.length == 0) 
		{
			document.getElementById("debugVertexId").value = vertexId;
			break;
		}
	}
}

function getTrailIds(endTrailId, traversal, backward) {
	let out = [];
	let trailId = endTrailId;
	while (trailId != -1) {
		out.push(trailId);
		trailId = traversal.prev_trail_ids[trailId];
	}
	out.reverse();

	let finalLocalState = {};
	if (endTrailId != -1) {
		for (k of localStateKeyOrder) {
			finalLocalState[k] = 0;
		}
	}
	for (trailId of out) {
		let localState = traversal.local_states[trailId];
		Object.assign(finalLocalState, localState);
	}

	if (backward) {
		out.reverse();
	}
	return [out, finalLocalState];
}

let localStateKeyOrder = [
	"energy_used",
	"reserves_used",
	"missiles_used",
	"supers_used",
	"power_bombs_used",
	"shinecharge_frames_remaining",
	"cycle_frames",
	"farm_baseline_energy_used",
	"farm_baseline_reserves_used",
	"farm_baseline_missiles_used",
	"farm_baseline_supers_used",
	"farm_baseline_power_bombs_used",
	"flash_suit",
];

function getDebugRoute(traversal, step, vertexId, costMetric, backward) {
	let traversalNumber;
	let endTrailId = -1;
	for (i in traversal.steps) {
		let s = traversal.steps[i];
		if (s.step_num > step) {
			break;
		}
		for (j in s.updated_vertex_ids) {
			if (s.updated_vertex_ids[j] == vertexId) {
				traversalNumber = i;
				endTrailId = s.updated_start_trail_ids[j][costMetric];
				break;
			}
		}
	}
	let [trailIdArray, finalLocalState] = getTrailIds(endTrailId, traversal, backward);

	let statePre = document.createElement("pre");
	if (Object.keys(finalLocalState).length > 0) {
		statePre.innerText = JSON.stringify(finalLocalState, localStateKeyOrder, 2);
	}

	let routeDiv = document.createElement("div");
	for (trailId of trailIdArray) {
		let linkIdx = traversal.link_idxs[trailId];
		let link = spoiler.game_data.links[linkIdx];
		let fromVertexId = link.from_vertex_id;
		let fromVertexKey = spoiler.game_data.vertices[fromVertexId];
		let fromNodeId = fromVertexKey.node_id;
		let toVertexId = link.to_vertex_id;
		let toVertexKey = spoiler.game_data.vertices[toVertexId];
		let roomId = toVertexKey.room_id;
		let toNodeId = toVertexKey.node_id;
		let stratId = link.strat_id;
		let stratName = link.strat_name;
		let room = roomMap[roomId];
		let node = nodeMap[[roomId, toNodeId]];
		let obstacleMask = toVertexKey.obstacle_mask;

		let mainLineDiv = document.createElement("div");
		let stratText = `[${toVertexId}] ${room.name}: ${node.name} (${obstacleMask}) {${linkIdx}} ${stratName}`;
		if (stratId !== null) {
			let mainLineA = document.createElement("a");
			mainLineA.innerText = stratText;
			mainLineA.href = `/logic/room/${roomId}/${fromNodeId}/${toNodeId}/${stratId}`;
			mainLineDiv.appendChild(mainLineA);
		} else {
			mainLineDiv.innerText = stratText;
		}
		routeDiv.appendChild(mainLineDiv);
		
		if (toVertexKey.actions.length > 0) {
			let vertexActionCode = document.createElement("code");
			vertexActionCode.innerText = JSON.stringify(toVertexKey.actions);
			routeDiv.appendChild(vertexActionCode);
		}

		let localState = traversal.local_states[trailId];
		if (Object.keys(localState).length > 0) {
			let localStateCode = document.createElement("code");
			localStateCode.innerText = JSON.stringify(localState);
			routeDiv.appendChild(localStateCode);	
		}
	}
	return [traversalNumber, statePre, routeDiv];
}

function updateDebugData() {
	let debugOutput = document.getElementById("debugOutput");
	debugOutput.innerHTML = "";
	let step = parseInt(document.getElementById("debugStepNumber").value);
	let details = spoiler.details[step];
	if (details === undefined) {
		return;
	}
	let vertexId = parseInt(document.getElementById("debugVertexId").value);
	if (vertexId === undefined || isNaN(vertexId)) {
		return;
	}
	if (vertexId < 0 || vertexId >= spoiler.game_data.vertices.length) {
		return;
	}
	let costMetric = parseInt(document.getElementById("debugCostMetric").value);
	if (costMetric < 0 || costMetric > 2) {
		return;
	}

	let vertexKey = spoiler.game_data.vertices[vertexId];
	let roomId = vertexKey.room_id;
	let nodeId = vertexKey.node_id;
	let roomName = roomMap[roomId].name;
	let nodeName = nodeMap[[roomId, nodeId]].name;
	let obstacleMask = vertexKey.obstacle_mask;

	let debugHeader = document.createElement("div");
	let headerMainLine = document.createElement("p");
	headerMainLine.innerText = `[${vertexId}] ${roomName}: ${nodeName} (${obstacleMask})`;
	debugHeader.appendChild(headerMainLine);
	if (vertexKey.actions.length > 0) {
		let actionPre = document.createElement("pre");
		actionPre.innerText = JSON.stringify(vertexKey.actions, null, 2);
		debugHeader.appendChild(actionPre);
	}
	debugOutput.appendChild(debugHeader);

	let [forwardTraversalNum, forwardState, forwardRoute] =
		getDebugRoute(spoiler.forward_traversal, step, vertexId, costMetric, false);
	let [reverseTraversalNum, reverseState, reverseRoute] =
		getDebugRoute(spoiler.reverse_traversal, step, vertexId, costMetric, true);
	
	let forwardStateDiv = document.createElement("div");
	let forwardStateHeader = createHtmlElement('<div class="category">OBTAIN STATE</div>');
	forwardStateDiv.appendChild(forwardStateHeader);
	forwardStateDiv.appendChild(forwardState);
	debugOutput.appendChild(forwardStateDiv);

	let reverseStateDiv = document.createElement("div");
	let reverseStateHeader = createHtmlElement('<div class="category">RETURN STATE</div>');
	reverseStateDiv.appendChild(reverseStateHeader);
	reverseStateDiv.appendChild(reverseState);
	debugOutput.appendChild(reverseStateDiv);

	let forwardRouteDiv = document.createElement("div");
	let forwardRouteHeader = createHtmlElement(`<div class="category">OBTAIN ROUTE (traversal number ${forwardTraversalNum})</div>`);
	forwardRouteDiv.appendChild(forwardRouteHeader);
	forwardRouteDiv.appendChild(forwardRoute);
	debugOutput.appendChild(forwardRouteDiv);

	let reverseDiv = document.createElement("div");
	let reverseHeader = createHtmlElement(`<div class="category">RETURN ROUTE (traversal number ${reverseTraversalNum})</div>`);
	reverseDiv.appendChild(reverseHeader);
	reverseDiv.appendChild(reverseRoute);
	debugOutput.appendChild(reverseDiv);

	debugOutput.style.paddingBottom = "16px";
}

document.getElementById("debugDataForm").addEventListener("submit", updateDebugData);
loadForm(document.getElementById("settingsForm"));
loadForm(document.getElementById("helpForm"));
setDebugDataVisibility();
if (!document.getElementById("showonce").checked)
	document.getElementById("msg-wrap").style.display = "flex";

let ctx = document.getElementById("spoiler-overlay").getContext("2d");
let grid = document.getElementById("grid");
let pat = ctx.createPattern(grid, "repeat");
ctx.fillStyle = pat;
ctx.fillRect(0,0,592,592);


fetch(`../spoiler.json`).then(c => c.json()).then(c => {
	spoiler = c;

	for (room of spoiler.game_data.rooms) {
		roomMap[room.room_id] = room;
	}
	for (node of spoiler.game_data.nodes) {
		nodeMap[[node.room_id, node.node_id]] = node;
	}

	flagtypes["objectives"] = c.objectives;
	flagtypes["objectives"].push("f_DefeatedMotherBrain");
	// generate map
	let map = new Array(72 * 72).fill(-1);
	let toiletTiles = new Array();
	let toiletbistep = -1;
	let toiletstep = -1;
	for (let i in c.all_rooms) {
		let v = c.all_rooms[i];
		for (let y = 0; y < v.map.length; y++) {
			for (let x = 0; x < v.map[y].length; x++) {
				if (v.map[y][x] != 0) {
					if (v.room == "Toilet") {
						if (toiletbistep < v.map_bireachable_step[y][x])
							toiletbistep = v.map_bireachable_step[y][x];
						if (toiletstep < v.map_bireachable_step[y][x])
							toiletstep = v.map_bireachable_step[y][x];
						toiletTiles.push((v.coords[1] + y) * 72 + (v.coords[0] + x));
					}
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
	function getRoomIndex(room_id) {
		for (i in c.all_rooms) {
			if (c.all_rooms[i].room_id == room_id) {
				return i;
			}
		}
		return null;
	}
	function addSuppItem(item,step, count, added_item)
	{
		let supp_div = document.getElementById("sidebar-supp-item");
		let ic = icon(item_plm[item.item]);
		ic.className = "ui-icon-hoverable";
		ic.id = item.item;
		ic.onclick = ev => {
			show_item_details(item.item, item.location, step, item);
		}
		supp_div.appendChild(ic);
		ic = document.createElement("span");
		ic.innerHTML = count;
		ic.classList.add("item-count");
		supp_div.appendChild(ic);
		if (added_item)
			return;

		let dblitem = item;
		supp_div.ondblclick = ev => {
			show_item_details(dblitem.item, dblitem.location, step, dblitem);
			ev.stopPropagation();
		}
	}
	function suppItems(step) {
		let supp_div = document.getElementById("sidebar-supp-item");
		let si = document.getElementById("sidebar-info");
		supp_div.style.display = "none";
		supp_div.innerHTML = "";
		
		if (!document.getElementById("spoilers").checked && step_limit < Number(step)+1)
			return;

		supp_div.style.left = si.offsetWidth+16+"px";
		supp_div.style.top = step * 24 +18+ "px";
		let items = c.details[step].items;
		let sortedItemIdxs = Array.from(items.keys()).sort((a, b) => item_rank[items[a].item] - item_rank[items[b].item]);
		let seen = new Set();
		let non_unique_counts = {
			"ETank": 1,
			"ReserveTank": 1,
			"Missile": 5,
			"PowerBomb": 5,
			"Super": 5
		};
		let last = null;
		let added_item = false;
		let count = 0;
		
		let ss = c.details[step].start_state.items;
		for (j of ss)
			seen.add(j);
		
		for (item_idx of sortedItemIdxs)
		{
			let j = items[item_idx];
			if (!non_unique_counts.hasOwnProperty(j.item))
				continue;
			if (last != null && last.item != j.item && count > 0)
			{
				addSuppItem(last,step, count, added_item);
				added_item = true;
				count = 0;
			}
			if (seen.has(j.item))
				count += non_unique_counts[j.item];
			else
				seen.add(j.item);
			last = j;
		}
		if (count != 0)
		{
			addSuppItem(last,step, count, added_item);
			added_item = true;
		}
		if (added_item)
			supp_div.style.display = "block";
	}
	let show_overview = () => {
		document.getElementById("path-overlay").innerHTML = ""
		let si = document.getElementById("sidebar-info");
		si.innerHTML = "";
		
		let seen = new Set();
		for (let i in c.summary) {
			let step_div = document.createElement("div");
			step_div.id = `step-${c.summary[i].step}`;
			step_div.className = "step-panel";
			step_div.onclick = () => {
				document.getElementById("path-overlay").innerHTML = "";
				gen_obscurity(c.summary[i].step);
				suppItems(i);
			}
			step_div.onmousemove = () => {
				suppItems(i);
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
					el.id = j.item;
					el.className = "ui-icon-hoverable";
					el.onclick = ev => {
						if (el.style.backgroundPositionX== `-${item_plm["Hidden"] * 16}px`)
						{
							gen_obscurity(Number(i)+1);
							suppItems(Number(i));
						}
						else
							show_item_details(j.item, j.location, i, j);
						ev.stopPropagation();
					}
					step_div.appendChild(el);

					if (first) {
						step_div.ondblclick = ev => {
							show_item_details(j.item, j.location, i, j);
							ev.stopPropagation();
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
		step_div.onclick = () => {
			document.getElementById("path-overlay").innerHTML = "";
			gen_obscurity(null);
		}

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
		step_div.appendChild(step_number);
		step_div.appendChild(help_button);

		si.appendChild(step_div);
		gen_obscurity();
	}
	window.gen_obscurity = (sl=step_limit) => {
		step_limit = sl;
		update_selected();
		let spoileron = document.getElementById("spoilers").checked;
		// generate obscurity+spoiler overlay
		let ov = document.getElementById("obscure-overlay");
		let ctx = ov.getContext("2d");
		let img = ctx.createImageData(72,72);
		let so = document.getElementById("spoiler-overlay");
		let sctx = so.getContext("2d");

		if (!spoileron) {
			while (document.getElementsByClassName("spoil").length > 0)
			{
				let e_spoils = document.getElementsByClassName("spoil");
				for (let e of e_spoils)
					while (e.classList.contains("spoil"))
						e.classList.remove("spoil");
			}
		}

		if (step_limit === null || spoileron)
			sl = c.summary.length;

		for (let i = 1;i<c.summary.length;i++) {
			let stepdiv = document.getElementById(`step-${i}`);
			if (stepdiv === null)
				continue;
			let items = stepdiv.getElementsByClassName("ui-icon-hoverable");
			for (let e of items)
			{
				if (!spoileron && i > sl)
					e.style.backgroundPositionX= `-${item_plm["Hidden"] * 16}px`;
				else
					e.style.backgroundPositionX= `-${item_plm[e.id] * 16}px`;
			}
		}

		for (let i=0;i<c.summary.length;i++) {
			if (i<sl) {
				for (let v of c.details[i].items) {
					let e = document.getElementById(v.location.room+": "+v.location.node);
					if (e) {
						e.style.backgroundPositionX= `-${item_plm[e.classList[0]] * 16}px`;
						if (!e.classList.contains("spoil"))
							e.classList.add("spoil");
					}
				}
				for (let v of c.details[i].flags){
					let e =document.getElementById(v.flag);
					if (e && !e.classList.contains("spoil"))
						e.classList.add("spoil");
				}
			} else {
				for (let v of c.details[i].items) {
					let e = document.getElementById(v.location.room+": "+v.location.node);
					if (e) {
						if (!spoileron && v.reachable_step > sl){
							e.style.backgroundPositionX= `-${item_plm["Hidden"] * 16}px`;
						}
						else {
							e.style.backgroundPositionX= `-${item_plm[e.classList[0]] * 16}px`;
						}
					}
				}
			}
		}

		let grid = document.getElementById("grid");
		let pat = ctx.createPattern(grid, "repeat");
		sctx.fillStyle = pat;
		sctx.fillRect(0,0,592,592);
		if (step_limit !== null && c.summary.length !=0) {
			for (let i = 0; i < 72 * 72; i++) {
				img.data[i * 4 + 3] = 0xd8; // Mostly opaque
			}
		}
		for (let v of c.all_rooms) {
			let explored = 0;
			let tiles = 0;
			let tilesExplored = new Array();
			for (let y = 0; y < v.map.length; y++) {
				for (let x = 0; x < v.map[y].length; x++) {
					if (v.map[y][x] == 1) {
						tiles++;
						let addr = (v.coords[1] + y) * 72 + (v.coords[0] + x);
						let sx = (v.coords[0] + x+1)*8;
						let sy = (v.coords[1] + y+1)*8;
						if (v.map_bireachable_step[y][x] < step_limit || step_limit === null
						   || (toiletTiles.includes(addr) && (toiletbistep < step_limit || toiletstep < step_limit)) ) {
							if (v.room == "Landing Site" && x==4 && y==4) {
								let e = document.getElementById("gunship");
								e.classList.add("spoil");
							}
							img.data[addr * 4 + 3] = 0x00; // transparent
							sctx.clearRect(sx,sy,8,8);
							explored++;
						} else if (v.map_reachable_step[y][x] < step_limit) {
							img.data[addr * 4 + 3] = 0x7F; // semiopaque
							sctx.clearRect(sx,sy,8,8);
							explored++;
						} else {
							if (spoileron) {
								sctx.clearRect(sx,sy,8,8);
								img.data[addr * 4 + 3] = 0xD8; // mostly opaque
							}
							else {
								tilesExplored.push([sx,sy]);
							}
						}
					}
				}
			}
			if (explored && explored != tiles && !spoileron) {
				let outline = document.getElementById("outline");
				for (let s of tilesExplored) {
					sctx.drawImage(outline, s[0],s[1], 8, 8, s[0], s[1], 8, 8);
				}
			}
		}
		ctx.putImageData(img, 0, 0);
	}
	let show_item_details = (item_name, loc, i, j, mapitem = false) => {
		if (j !== null) {
			document.getElementById("path-overlay").innerHTML = ""
			showRoute(j.return_route, "yellow");
			showRoute(j.obtain_route);
			document.getElementById("sidebar-supp-item").style.display = "none";
		}
		let si = document.getElementById("sidebar-info");
		si.scrollTop = 0;
		si.innerHTML = "";
		if (j !== null) {
			if (!mapitem)
				step_limit = c.details[i].step;
			else if (c.details[i].step > step_limit)
				step_limit = c.details[i].step;
			let title = document.createElement("div");
			title.className = "sidebar-title";
			title.innerHTML = `STEP ${c.details[i].step}`;
			si.appendChild(title);
		}

		if (j !== null) {
			gen_obscurity(step_limit);

			let previous_header = document.createElement("div");
			previous_header.className = "category";
			previous_header.innerHTML = "PREVIOUSLY COLLECTIBLE";
			si.appendChild(previous_header);

			let ss = c.details[i].start_state;
			flagIcons(si, ss.flags);
			
			let non_unique_item_list = document.createElement("div");
			non_unique_item_list.className = "item-list";
			let s = [ss.max_missiles, ss.max_supers, ss.max_power_bombs, Math.floor(ss.max_energy / 100), ss.max_reserves / 100];
			let co = [ss.collectible_missiles, ss.collectible_supers, ss.collectible_power_bombs, null, null];
			let ic = [1, 2, 3, 0, 20];
			for (let i in s) {
				if (s[i] > 0) {
					non_unique_item_list.appendChild(icon(ic[i]));
					let count = document.createElement("span");
					if (co[i] !== null) {
						count.innerHTML = s[i] + " / " + co[i] + " ";
					} else {
						count.innerHTML = s[i] + " ";
					}
					non_unique_item_list.appendChild(count);
				}
			}
			si.appendChild(non_unique_item_list);

			let unique_item_list = document.createElement("div");
			unique_item_list.className = "item-list";
			for (let i of ss.items) {
				if (i == "Nothing") { continue; }
				if (!ic.includes(item_plm[i])) {
					unique_item_list.appendChild(icon(item_plm[i]));
				}
			}
			si.appendChild(unique_item_list);
			

			let collectible_header = document.createElement("div");
			collectible_header.className = "category";
			collectible_header.innerHTML = "COLLECTIBLE ON THIS STEP";
			si.appendChild(collectible_header);

			if (i !== null)
				flagIcons(si, c.summary[i].flags, j);

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
				icon_el.onmouseenter = ev => {
					document.getElementById(item.location.room +": "+ item.location.node).classList.add("highlight");
				}
				icon_el.onmouseleave = ev => {
					document.getElementById(item.location.room +": "+ item.location.node).classList.remove("highlight");
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
			routeData(item_info, j.obtain_route, ss);
				
			if (j.return_route.length !=0){
				item_info.appendChild(createHtmlElement(`<div class="category">RETURN ROUTE</div>`));
				routeData(item_info, j.return_route);
			} else {
				let escText = createDiv("ESCAPE");
				if (c.escape.animals_route){
					escText.innerHTML += " WITH THE ANIMALS";
				}
				escText.className = "category";
				item_info.appendChild(escText);
			}
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
	
	function highlightRoute(to, room_id, iters) {
		if (to == null || to.length == 0)
			return;

		path = "";
		found = 0;
		sameroom = false;
		for (let k of to) {
			if (k.room_id == room_id) {
				if (found != iters && !sameroom) {
					found++;
					sameroom = true;
				}
				if  (found == iters && k.coords) {
					let x = k.coords[0] * 24 + 24 + 12;
					let y = k.coords[1] * 24 + 24 + 12;
					path += `${path == "" ? "M" : "L"}${x} ${y} `;
				}
			} else if (found == iters) {
				if  (k.coords) {
					let x = k.coords[0] * 24 + 24 + 12;
					let y = k.coords[1] * 24 + 24 + 12;
					path += `${path == "" ? "M" : "L"}${x} ${y} `;
				}
				break;
			} else {
				sameroom = false;
			}
		}
		document.getElementById("path-highlight").innerHTML += `<path d="${path}" id="path-out"/>`
		document.getElementById("path-highlight").innerHTML += `<path d="${path}" id="path-in"/>`
	}
	function routeData(p, route, ss=null) {
		let lastRoom=null, lastNode=null, roomDiv=null, roomRoute=null;
		let room_reps = new Map();
		for (let k of route) {
			let strat_url = `/logic/room/${k.room_id}/${k.from_node_id}/${k.to_node_id}/${k.strat_id}`;
			let nodeStr;
			let out = "";
			if (k.room != lastRoom) {
				if (roomDiv) {
					p.appendChild(roomDiv);
				}
				
				if (!room_reps.has(k.room_id)){
					room_reps.set(k.room_id, 1);
				} else {
					room_reps.set(k.room_id, room_reps.get(k.room_id)+1);
				}

				let rr = document.createElement("div");
				rr.className = "room-route";

				let roomHead = document.createElement("span");
				roomHead.innerHTML = `${k.room}`;
				roomHead.className = "room-head";

				let arrow = document.createElement("i");
				arrow.className="bi bi-arrow-right";
				roomHead.appendChild(arrow);

				roomHead.onclick = ev => {
					if (rr.style.display == "block") {
						arrow.className="bi bi-arrow-right";
						rr.style.display = "none";
					} else {
						arrow.className="bi bi-arrow-down";
						rr.style.display = "block";
					}
				}
				roomRoute = rr;
				
				roomDiv = document.createElement("div");
				let reps = room_reps.get(k.room_id);
				roomDiv.onmouseenter = ev => {
					highlightRoute(route,k.room_id, reps);
				}
				roomDiv.onmouseleave = ev => {
					document.getElementById("path-highlight").innerHTML = "";
				}
				roomDiv.appendChild(roomHead);
				roomDiv.appendChild(roomRoute);
			}
			if (ss == null)	{
				out = consumableData(k);
				if (out != "") {
					let cons = document.createElement("div");
					cons.className = "route-consumables";
					cons.innerHTML = `<small>${out}</small>`;
					roomRoute.appendChild(cons);
				}
			}

			if (k.strat_id !== null) {
				nodeStr = `<a class="room-link" href="${strat_url}">${k.room}: ${k.node}</a><br>`;
			} else {
				nodeStr = `${k.room}: ${k.node}<br>`;
			}
			if (k.room != lastRoom || k.node != lastNode || k.strat_id !== null) {
				let node_div = createDiv(nodeStr);
				roomRoute.appendChild(node_div);
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
				let strat = document.createElement("div");
				strat.className = "route-strat";
				strat.innerHTML = `<small>${out}</small>`;
				roomRoute.appendChild(strat);
			}
			if (ss != null)	{
				out = consumableData(k, ss);
				if (out != "") {
					let cons = document.createElement("div");
					cons.className = "route-consumables";
					cons.innerHTML = `<small>${out}</small>`;
					roomRoute.appendChild(cons);
				}
			}
			
			if (k.relevant_flags) {
				let flagDiv = document.createElement("div");
				flagDiv.className = "route-flags";
				let flagContainer = document.createElement("small");
				flagDiv.appendChild(flagContainer);
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
				roomRoute.appendChild(flagDiv);
			}
		}
		p.appendChild(roomDiv);
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
			e.onmouseenter = ev => {
				document.getElementById(f).classList.add("highlight");
			}
			
			e.onmouseleave = ev => {
				document.getElementById(f).classList.remove("highlight");
			}
			p.appendChild(e);
		}
	}
	function showFlag(details, flagName, mapflag=false) {
		for (let stepNum in details) {
			let stepData = details[stepNum];
			for (let flagData of stepData.flags) {
				if (flagData.flag == flagName) {
					show_item_details(flagName, flagData.location, stepNum, flagData, mapflag);
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

		if (c.hub_obtain_route == null || c.hub_return_route == null)
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
		routeData(item_info, c.hub_obtain_route, ss);
		
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
			i = getRoomIndex(8); // Landing Site
			x = c.all_rooms[i].coords[0]*24+24;
			y = c.all_rooms[i].coords[1]*24+24;
			x += 96;
			y += 72;	
		} else if (n == "Bomb Torizo Room") {
			startitems=3;
		} else if (n == "") {
			// escape
			n = "Mother Brain Room";
			i = getRoomIndex(238);  // Mother Brain Room
			x = c.all_rooms[i].coords[0]*24+24;
			y = c.all_rooms[i].coords[1]*24+24;
		} else if (n == "Homing Geemer Room") {
			i = getRoomIndex(32);  // West Ocean
			x = c.all_rooms[i].coords[0]*24+24;
			y = c.all_rooms[i].coords[1]*24+24;
			x += 120;
			y += 48;
		} else if (n == "East Pants Room") {
			i = getRoomIndex(220);  // Pants Room
			x = c.all_rooms[i].coords[0]*24+24;
			y = c.all_rooms[i].coords[1]*24+24;
			x += 24;
			y += 24;
		}

		sr = c.all_rooms[i];
		startbase.x = x;
		startbase.y = y;
		for (i in c.all_items) {
			let loc = c.all_items[i].location;
			if (loc.room_id == ri) {
				let os = lookupOffset(loc.room_id, loc.node_id);
				let lx = loc.coords[0]*24 + 24;
				let ly = loc.coords[1]*24 + 24;
				if (os) {
					lx += Math.round(os[0])*24;
					ly += Math.round(os[1])*24;
				}
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
		if (c.summary.length == 0) {
			e.onclick = ev => {
				showEscape();
			}
		}
		else {
			e.onclick = ev => {
				hubRoute();
			}
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
		sr = c.all_rooms[getRoomIndex(8)];  // Landing Site
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
			let reach_step = -1;
			for (v in c.details){
				for (let vf of c.details[v].flags){
					if (vf.flag == "f_DefeatedMotherBrain"){
						reach_step = Number(vf.reachable_step);
						break;
					}
				}
			}
			if (!document.getElementById("spoilers").checked && step_limit != null &&  reach_step > step_limit)
			{
				document.getElementById("shipspoiler").style.display = "block"
				
				setTimeout(fn => {document.getElementById("shipspoiler").style.display = "none";}, 1000)
				return;
			}

			step_limit = null;
			
			document.getElementById("path-overlay").innerHTML = ""
			show_overview();
			update_selected();
			showEscape();
			gen_obscurity();
		}
		e.onpointermove = ev => {
			hideRoom();
		}
		document.getElementById("overlay").appendChild(e);
		e = document.createElement("div");
		e.className = "popup";
		e.innerHTML = `<b>Ship</b><br><small>${sr.room}</small><br><div id="shipspoiler" style="display:none"><small>Escape not in logic on this step</small></div>`;
		e.style.left = x + 48 +"px";
		e.style.top = y + "px";
		document.getElementById("overlay").appendChild(e);
	}

	flags: 
	if (c.summary.length != 0) {
		for (i in roomFlags) {
			e = document.createElement("img");
			let rf = roomFlags[i];
			let f = rf[0];
			let obj = false;
			if (f == "f_ZebesAwake")
				continue;


			var found = false;
			for (j in c.all_rooms)	{
				if (c.all_rooms[j].room_id == i) {
					found = true;
					break;
				}
			}
			if (!found) {
				continue;
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

			let reach_step = -1;
			let v = -1;
			for (v in c.details){
				for (let vf of c.details[v].flags){
					if (vf.flag == f){
						reach_step = Number(vf.reachable_step);
						break;
					}
				}
				if (reach_step != -1)
					break;
			}
			e.onclick = ev => {
				if (document.getElementById("spoilers").checked || document.getElementById("spoilers").checked || step_limit === null || step_limit > v)
					showFlag(c.details, f, true);
			}
			e.onpointermove = ev => {
				hideRoom();
				if (!document.getElementById("spoilers").checked && step_limit !== null && step_limit >= reach_step && step_limit <= Number(v)) {
					el.innerHTML = `<b>${rf[1]}</b><br><small>${sr.room}</small><br>Not in logic for current step.`;
					el.style.left = Number(ev.target.style.left.substring(0,ev.target.style.left.length-2))+16+"px";
					el.style.top = ev.target.style.top;
					el.classList.remove("hidden");
				}
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
			let os = lookupOffset(v.location.room_id, v.location.node_id);
			if (os) {
				v.location.coords[0] += os[0];
				v.location.coords[1] += os[1];
			}
			e = document.createElement("div");
			e.className = v.item;
			e.classList.add("icon");
			e.classList.add("items");
			
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
				if (i !== null)
					break;
			}
			let reach_step = j !== null ? j.reachable_step : null;
			let step = Number(i);
			e.onclick = ev => {
				if (document.getElementById("spoilers").checked || step_limit === null || step_limit > i) {
					show_item_details(v.item, v.location, i, j, true);
				}
			};
			e.onpointermove = ev => {
				hideRoom();
				if (!document.getElementById("spoilers").checked && step_limit !== null && step_limit <= step && step_limit >= reach_step) {
					el.innerHTML = `<b>${v.item}</b><br><small>${v.location.room}</small><br>Not in logic on this step`;
					el.style.left = Number(ev.target.style.left.substring(0,ev.target.style.left.length-2))+16+"px";
					el.style.top = ev.target.style.top;
					el.classList.remove("hidden");
				}
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
	moveStart();
	show_overview();
	if (c.summary.length == 0)
	{
		gen_obscurity(null);
		showEscape();
	}
	else
		gen_obscurity(1);
	
	document.getElementById("map").style.visibility = "visible";

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
				let v = c.all_rooms[tile];
				let i = y -v.coords[1];
				let j = x -v.coords[0];
				if (step_limit != null && v.map_reachable_step[i][j] >= step_limit) {
					el.classList.add("hidden");
					el.innerText = "";
					return;
				}
				el.innerText = v.room;
				el.dataset.roomId = v.room_id;
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
			update_selected();
			document.getElementById("path-overlay").innerHTML = "";
			document.getElementById("debugOutput").innerHTML = "";
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
		document.getElementById("sidebar-supp-item").style.display = "none";
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
