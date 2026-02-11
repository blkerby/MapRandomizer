#!/bin/bash
# Rebase the map-rando branch on top of the latest upstream master (or specified commit),
# squashing the changes since the previous rebase (marked by the "base" tag).
#
# Usage: ./scripts/rebase_sm_json.sh [commit]
#
# See https://amboar.github.io/notes/2021/09/16/history-preserving-fork-maintenance-with-git.html
# for the idea behind this approach.
set -eux
COMMIT=${1:-upstream/master}
cd sm-json-data
git fetch upstream
git fetch kjbranch
git checkout ${COMMIT}

# Retain a reference to the existing map-rando branch as a second parent (ignored in the merge):
# This preserves the history of the map-rando branch even while we rebase it.
git merge -s ours kjbranch/map-rando -m "New upstream version"
git tag new-base

# Squash the changes since the previous rebase into a single commit.
# Exclude the auto-generated schema files from the commit, to avoid potential unnecessary conflicts.
git checkout kjbranch/map-rando
git reset --soft base
git add .
git restore schema/m3-string-requirements.schema.json
git restore schema/m3-numeric-parameters.schema.json
git commit -am "Map Rando overrides"

# Rebase onto the new base, run the tests, and amend the commit to update auto-generated schema files:
git rebase --onto new-base base HEAD
bash resources/ci/common/sh/test_python.sh
git commit --amend -a --no-edit

# Move the "base" tag forward to the new base, and push the rebased branch.
# Note we only force push the "base" tag, not the "map-rando" branch itself.
git tag -f base new-base
git tag -d new-base
git push kjbranch HEAD:map-rando
git push -f kjbranch base
git checkout kjbranch/map-rando
