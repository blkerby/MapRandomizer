"""Print a Logic updates HTML block comparing working-tree data with v{VERSION-1}.

Run from any directory. Empty sections and currently Uncategorized entries are
omitted; entries moved from Uncategorized or Ignored count as new.
"""

import html
import json
from pathlib import Path
import subprocess
import sys


DIFFICULTIES = [
    "Implicit", "Basic", "Medium", "Hard", "Very Hard", "Expert", "Expert+",
    "Extreme", "Extreme+", "Insane", "Insane+", "Beyond", "Ignored",
]


def load_previous(repo, tag, path):
    result = subprocess.run(
        ["git", "show", f"{tag}:{path}"], cwd=repo,
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def entry_key(entry, kind):
    if kind == "tech":
        return (entry["tech_id"],)
    return (entry["room_id"], entry["notable_id"])


def entry_link(entry, kind):
    ids = "/".join(str(value) for value in entry_key(entry, kind))
    name = entry["name"]
    if kind == "notable":
        name = f'{entry["room_name"]}: {name}'
    return f'<a href="/logic/{kind}/{ids}">{html.escape(name)}</a>'


def make_sections(current, previous, kind):
    previous_by_id = {entry_key(entry, kind): entry for entry in previous}
    new_entries = []
    moved_entries = []
    # Sort by current difficulty, preserving source order within each tier.
    difficulty_order = {name: index for index, name in enumerate(DIFFICULTIES)}
    current = sorted(
        current,
        key=lambda entry: difficulty_order.get(entry["difficulty"], len(DIFFICULTIES)),
    )
    for entry in current:
        difficulty = entry["difficulty"]
        if difficulty == "Uncategorized":
            continue
        previous_entry = previous_by_id.get(entry_key(entry, kind))
        link = entry_link(entry, kind)
        if (
            previous_entry is None
            or previous_entry["difficulty"] == "Uncategorized"
            or (previous_entry["difficulty"] == "Ignored" and difficulty != "Ignored")
        ):
            new_entries.append(f"{link} ({html.escape(difficulty)}).")
        elif previous_entry["difficulty"] != difficulty:
            old_difficulty = html.escape(previous_entry["difficulty"])
            moved_entries.append(
                f"{link} is moved from {old_difficulty} to {html.escape(difficulty)}."
            )
    return new_entries, moved_entries


def render_section(title, entries):
    if not entries:
        return []
    return [
        f"    <li>{title}:",
        "        <ul>",
        *(f"        <li>{entry}" for entry in entries),
        "        </ul>",
    ]


def main():
    repo = Path(__file__).resolve().parents[2]
    version = int((repo / "rust/VERSION").read_text().strip())
    tag = f"v{version - 1}"
    sections = {}
    for kind, filename in [("tech", "tech_data.json"), ("notable", "notable_data.json")]:
        path = f"rust/data/{filename}"
        current = json.loads((repo / path).read_text())
        previous = load_previous(repo, tag, path)
        sections[kind] = make_sections(current, previous, kind)

    lines = []
    for title, entries in [
        ("New tech", sections["tech"][0]),
        ("New notables", sections["notable"][0]),
        ("Tech moved to a different difficulty level", sections["tech"][1]),
        ("Notables moved to a different difficulty level", sections["notable"][1]),
    ]:
        lines.extend(render_section(title, entries))
    if lines:
        print("\n".join(["<li>Logic updates:", "    <ul>", *lines, "    </ul>"]))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Unable to read previous release data: {exc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Unable to generate logic release notes: {exc}", file=sys.stderr)
        sys.exit(1)
