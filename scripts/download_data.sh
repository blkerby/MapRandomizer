#!/bin/bash
set -e

mkdir -p tmp
cd tmp

# Download the map pools
mkdir -p ../maps
for pool in "v113-tame" "v110c-wild"
do
wget https://storage.googleapis.com/super-metroid-map-rando/maps/${pool}.tgz
tar xfz ${pool}.tgz --directory ../maps
rm ${pool}.tgz
done

# Download the Mosaic patches
export MOSAIC_COMMIT_ID=$(cat ../MOSAIC_COMMIT_ID)
wget https://map-rando-artifacts.s3.us-west-004.backblazeb2.com/Mosaic/Mosaic-${MOSAIC_COMMIT_ID}.tar.zstd
unzstd Mosaic-${MOSAIC_COMMIT_ID}.tar.zstd
mkdir -p ../patches/mosaic
tar xf Mosaic-${MOSAIC_COMMIT_ID}.tar --directory ../patches/mosaic
rm Mosaic-${MOSAIC_COMMIT_ID}.tar Mosaic-${MOSAIC_COMMIT_ID}.tar.zstd
