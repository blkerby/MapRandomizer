#!/bin/sh
set -e

cd -- "$(dirname -- "$0")/.."

git submodule update --init -- sm-json-data
git -C sm-json-data fetch https://github.com/kjbranch/sm-json-data.git map-rando
git -C sm-json-data checkout --detach FETCH_HEAD

export PYTHONPATH=python
python python/scripts/update_tech.py
python python/scripts/update_notables.py
python python/scripts/update_presets.py
python python/scripts/update_video_listing.py
