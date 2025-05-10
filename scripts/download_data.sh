#!/bin/bash
set -e

mkdir -p tmp
cd tmp

# Download the map pools
mkdir -p ../maps
for pool in "v117c-standard" "v117c-wild"
do
wget https://map-rando-artifacts.s3.us-west-004.backblazeb2.com/maps/${pool}.tgz
tar xfz ${pool}.tgz --directory ../maps
rm ${pool}.tgz
done

# Download the Mosaic patches
export MOSAIC_BUILD_ID=$(cat ../MOSAIC_BUILD_ID)
wget https://map-rando-artifacts.s3.us-west-004.backblazeb2.com/Mosaic/Mosaic-${MOSAIC_BUILD_ID}.tar.zstd
zstd -d Mosaic-${MOSAIC_BUILD_ID}.tar.zstd -o Mosaic-${MOSAIC_BUILD_ID}.tar
mkdir -p ../patches/mosaic
tar xf Mosaic-${MOSAIC_BUILD_ID}.tar --directory ../patches/mosaic
rm Mosaic-${MOSAIC_BUILD_ID}.tar Mosaic-${MOSAIC_BUILD_ID}.tar.zstd
