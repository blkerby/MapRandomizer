#!/bin/bash
set -e

# `wget` writes progress to stderr; suppress it outside a terminal (e.g. in CI).
wget_args=()
if [[ ! -t 2 ]]; then
    wget_args+=(--no-verbose)
fi

mkdir -p tmp
cd tmp

# Download the map pools
mkdir -p ../maps
for pool in "v119-small-avro" "v119-standard-avro" "v119-wild-avro"
do
wget "${wget_args[@]}" https://map-rando-artifacts.s3.us-west-004.backblazeb2.com/maps/${pool}.tar
tar xf ${pool}.tar --directory ../maps
rm ${pool}.tar
done

# Download the Mosaic patches
export MOSAIC_BUILD_ID=$(cat ../MOSAIC_BUILD_ID)
wget "${wget_args[@]}" https://map-rando-artifacts.s3.us-west-004.backblazeb2.com/Mosaic/Mosaic-${MOSAIC_BUILD_ID}.tar.zstd
zstd -d Mosaic-${MOSAIC_BUILD_ID}.tar.zstd -o Mosaic-${MOSAIC_BUILD_ID}.tar
mkdir -p ../patches/mosaic
tar xf Mosaic-${MOSAIC_BUILD_ID}.tar --directory ../patches/mosaic
rm Mosaic-${MOSAIC_BUILD_ID}.tar Mosaic-${MOSAIC_BUILD_ID}.tar.zstd
