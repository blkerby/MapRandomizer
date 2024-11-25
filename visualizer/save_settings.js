function saveForm(form) {
    data = {}
    for (const element of form.elements) {
        if (element.type == "file") {
            continue;
        }
        if (element.type == "radio" && !element.checked) {
            if (data[element.name] === undefined) {
                data[element.name] = '';
            }
            continue;
        }
        if (element.name == "") {
            continue;
        }
        if (element.type == "checkbox") {
            data[element.name] = element.checked;
        } else {
            data[element.name] = element.value;
        }
    }
    localStorage[form.id] = JSON.stringify(data);
}

function migrateValue(name, value) {
    var migrationMapping = {
        "maps_revealed": {
            "false": "No",
            "true": "Full",
            "Yes": "Full"
        },
        "door_locks_size": {
            "small": "Small",
            "large": "Large",
        },
        "music": {
            "vanilla": "area"
        },
        "samus_sprite": {
            "samus": "samus_vanilla"
        },
        "random_tank": {
            "Yes": "true",
            "No": "false"
        },
        "spazer_before_plasma": {
            "Yes": "true",
            "No": "false"
        },
        "race_mode": {
            "Yes": "true",
            "No": "false"
        },
        "stop_item_placement_early":  {
            "Yes": "true",
            "No": "false"
        },
    };
    if (migrationMapping[name] !== undefined) {
        var newValue = migrationMapping[name][value];
        if (newValue !== undefined) {
            return newValue;
        }
    }
    return value;
}

function loadForm(form) {
    if (localStorage[form.id] === undefined) {
        return;
    }
    data = JSON.parse(localStorage[form.id]);
    for (const element of form.elements) {
        if (element.type == "file") {
            continue;
        }
        if (element.name == "") {
            continue;
        }
        if (element.type == "radio") {
            var value = migrateValue(element.name, data[element.name]);
            if (value == element.value) {
                element.checked = true;
            } else if (value !== undefined) {
                element.checked = false;
            }
        } else if (data[element.name] !== undefined) {
            if (element.type == "checkbox") {
                element.checked = migrateValue(element.name, data[element.name]);
            } else {
                element.value = migrateValue(element.name, data[element.name]);
            }
        }
    }
}