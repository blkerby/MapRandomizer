set -e
cd sm-json-data
git fetch origin
git checkout origin/master
cd ..
export PYTHONPATH=python
python python/scripts/update_tech.py
python python/scripts/update_notables.py
python python/scripts/update_presets.py
python python/scripts/update_video_listing.py
