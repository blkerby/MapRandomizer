#!/bin/bash
# Usage: ./scripts/finish_rebase_sm_json.sh
set -eux
cd sm-json-data
bash resources/ci/common/sh/test_python.sh
git commit --amend -a --no-edit

# Move the "base" tag forward to the new base, and push the rebased branch.
# Note we only force push the "base" tag, not the "map-rando" branch itself.
git tag -f base new-base
git tag -d new-base
git push kjbranch HEAD:map-rando
git push -f kjbranch base
git checkout kjbranch/map-rando
