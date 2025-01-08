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
echo Mosaic commit: $MOSAIC_COMMIT_ID

cd rust
cargo run --release --bin build-mosaic --  --compressor ~/bin/AmoebaCompress --input-rom ../roms/vanilla.sfc
cd ..

cd patches/mosaic
tar cf - . | zstd -19 >../../tmp/Mosaic.tar.zstd
cd ../..
source ~/credentials/backblaze-map-rando-artifacts.sh
aws s3 cp tmp/Mosaic.tar.zstd s3://map-rando-artifacts/Mosaic/Mosaic-${MOSAIC_COMMIT_ID}.tar.zstd --endpoint ${AWS_ENDPOINT}

echo -n $MOSAIC_COMMIT_ID >MOSAIC_COMMIT_ID
