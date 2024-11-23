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

declare -a patches=(
    "Area FX"
    "Area Palettes"
    "Area Palette Glows"
    "Bowling"
    "Scrolling Sky v1.6"
)

for PATCH in "${patches[@]}"
do
    echo ${PATCH}
    cp roms/vanilla.sfc roms/tmp-rom.sfc
    asar "Mosaic/Projects/Base/ASM/${PATCH}.asm" roms/tmp-rom.sfc
    ips_util create roms/vanilla.sfc roms/tmp-rom.sfc >"patches/ips/${PATCH}.ips"
done

cd rust
cargo run --release --bin build-mosaic --  --compressor ~/bin/AmoebaCompress --input-rom ../roms/vanilla.sfc
cd ..

cd patches/mosaic
tar cf - . | zstd -19 >../../tmp/Mosaic.tar.zstd
cd ../..
source ~/credentials/backblaze-map-rando-artifacts.sh
aws s3 cp tmp/Mosaic.tar.zstd s3://map-rando-artifacts/Mosaic/Mosaic-${MOSAIC_COMMIT_ID}.tar.zstd --endpoint ${AWS_ENDPOINT}

echo -n $MOSAIC_COMMIT_ID >MOSAIC_COMMIT_ID
