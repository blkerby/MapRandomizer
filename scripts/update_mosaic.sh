#!/bin/bash

set -e

if [ ${SKIP_CHECKOUT:-0} -ne 1 ]; then
echo Checking out latest commit
cd Mosaic
git fetch origin
git checkout origin/main
cd ..
else
echo Skipping checkout
fi

export MOSAIC_COMMIT_ID=$(git -C Mosaic rev-parse HEAD)
export MOSAIC_BUILD_ID=$(uuidgen)
echo Mosaic commit: $MOSAIC_COMMIT_ID
echo Mosaic build ID: $MOSAIC_BUILD_ID

cd rust
# cargo run --release --bin build-mosaic --  --compressor ~/bin/AmoebaCompress --input-rom ../roms/vanilla.sfc
# using compressor by NobodyNada, with tweaks to help give faster decompression
cargo run --release --bin build-mosaic -- --input-rom ../roms/vanilla.sfc
cd ..

cd patches/mosaic
tar cf - . | zstd -19 >../../tmp/Mosaic.tar.zstd
cd ../..
source ~/credentials/backblaze-map-rando-artifacts.sh
aws s3 cp tmp/Mosaic.tar.zstd s3://map-rando-artifacts/Mosaic/Mosaic-${MOSAIC_BUILD_ID}.tar.zstd --endpoint ${AWS_ENDPOINT}

echo -n $MOSAIC_BUILD_ID >MOSAIC_BUILD_ID
