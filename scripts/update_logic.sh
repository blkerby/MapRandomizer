#!/bin/sh
set -e

for arg in "$@"; do
    case "$arg" in
        --strict) ;;
        *) echo "Usage: $0 [--strict]" >&2; exit 1 ;;
    esac
done

cd -- "$(dirname -- "$0")/.."

git submodule update --init -- sm-json-data
git -C sm-json-data fetch https://github.com/kjbranch/sm-json-data.git map-rando
git -C sm-json-data checkout --detach FETCH_HEAD

export PYTHONPATH=python
python python/scripts/update_tech.py "$@"
python python/scripts/update_notables.py "$@"
python python/scripts/update_presets.py
python python/scripts/update_video_listing.py
