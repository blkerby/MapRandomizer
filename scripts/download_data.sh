#!/bin/bash
set -e

mkdir -p tmp
cd tmp

# Download the map pools
mkdir -p ../maps
for pool in "v119-standard-avro" "v119-wild-avro"
do
wget https://map-rando-artifacts.s3.us-west-004.backblazeb2.com/maps/${pool}.tar
tar xf ${pool}.tar --directory ../maps
rm ${pool}.tar
done

# Download the Mosaic patches
export MOSAIC_BUILD_ID=$(cat ../MOSAIC_BUILD_ID)
wget https://map-rando-artifacts.s3.us-west-004.backblazeb2.com/Mosaic/Mosaic-${MOSAIC_BUILD_ID}.tar.zstd
zstd -d Mosaic-${MOSAIC_BUILD_ID}.tar.zstd -o Mosaic-${MOSAIC_BUILD_ID}.tar
mkdir -p ../patches/mosaic
tar xf Mosaic-${MOSAIC_BUILD_ID}.tar --directory ../patches/mosaic
rm Mosaic-${MOSAIC_BUILD_ID}.tar Mosaic-${MOSAIC_BUILD_ID}.tar.zstd
