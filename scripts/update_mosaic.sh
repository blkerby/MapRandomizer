#!/bin/sh
( cd rust && cargo run --bin compress-retiling -- --compressor /home/kerby/Downloads/AmoebaCompress )
diff -r Mosaic/Projects/Base/ASM patches/src/Mosaic

