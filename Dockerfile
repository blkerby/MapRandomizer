FROM rust:1.83.0-bullseye AS build

RUN apt-get update && apt-get install -y zstd

# First get Cargo to download the crates.io index (which takes a long time) via `cargo install lazy_static`
# Both `cargo update` and `crater search` no longer update the crates.io index, see: https://github.com/rust-lang/cargo/issues/3377
RUN cargo new --bin rust
WORKDIR /rust
RUN cargo install lazy_static; exit 0

# Now use a dummy binary to build the project dependencies (allowing the results to be cached)
COPY rust/Cargo.lock /rust/Cargo.lock
COPY rust/Cargo.toml /rust/Cargo.toml
COPY rust/maprando/Cargo.toml /rust/maprando/Cargo.toml
COPY rust/maprando/src/bin/dummy.rs /rust/maprando/src/bin/dummy.rs
COPY rust/maprando-web/Cargo.toml /rust/maprando-web/Cargo.toml
COPY rust/maprando/src/bin/dummy.rs /rust/maprando-web/src/bin/dummy-web.rs
COPY rust/maprando-game/Cargo.toml /rust/maprando-game/Cargo.toml
COPY rust/maprando/src/bin/dummy.rs /rust/maprando-game/src/bin/dummy-game.rs
COPY rust/maprando-logic/Cargo.toml /rust/maprando-logic/Cargo.toml
COPY rust/maprando/src/bin/dummy.rs /rust/maprando-logic/src/bin/dummy-logic.rs
COPY rust/maprando-wasm/Cargo.toml /rust/maprando-wasm/Cargo.toml
COPY rust/maprando/src/bin/dummy.rs /rust/maprando-wasm/src/bin/dummy-wasm.rs
RUN mkdir -p /rust/maprando-wasm/src && touch /rust/maprando-wasm/src/lib.rs
RUN cargo build --release
RUN rm /rust/src/*.rs

# Download the map datasets and Mosaic patches
WORKDIR /
COPY /scripts /scripts
COPY /MOSAIC_COMMIT_ID /MOSAIC_COMMIT_ID
RUN bash /scripts/download_data.sh

# Now copy over the source code and build the real binary
RUN cargo install wasm-pack
COPY rust /rust
WORKDIR /rust/maprando-wasm
RUN wasm-pack build --target="web" --release
WORKDIR /rust
RUN cargo build --release --bin maprando-web

# Now restart with a slim base image and just copy over the binary and data needed at runtime.
FROM debian:buster-slim
RUN apt-get update && apt-get install -y \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*
COPY maps/vanilla /maps/vanilla
COPY patches /patches
COPY gfx /gfx
COPY sm-json-data /sm-json-data
COPY MapRandoSprites /MapRandoSprites
COPY TitleScreen /TitleScreen
COPY room_geometry.json /
COPY palette_smart_exports /palette_smart_exports
COPY visualizer /visualizer
# Both stages will run in parallel until the build stage is referenced,
# at which point this stage will wait for the `build` stage to complete, so delay these until last
COPY --from=build /maps /maps
COPY --from=build /patches/mosaic /patches/mosaic
COPY --from=build /rust/data /rust/data
COPY --from=build /rust/static /rust/static
COPY --from=build /rust/maprando-wasm/pkg /rust/maprando-wasm/pkg
# Since the bin is the most likely thing to have changed, copy it last to avoid invalidating the rest of the steps
COPY --from=build /rust/target/release/maprando-web /rust
WORKDIR /rust
ENTRYPOINT ["/rust/maprando-web"]
