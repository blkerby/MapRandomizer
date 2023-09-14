FROM rust:1.67.0-buster as build

# First get Cargo to download the crates.io index (which takes a long time) via `cargo install lazy_static`
# Both `cargo update` and `crater search` no longer update the crates.io index, see: https://github.com/rust-lang/cargo/issues/3377
RUN cargo new --bin rust
WORKDIR /rust
RUN cargo install lazy_static; exit 0

# Now use a dummy binary to build the project dependencies (allowing the results to be cached)
COPY rust/Cargo.lock /rust/Cargo.lock
COPY rust/Cargo.toml /rust/Cargo.toml
RUN cargo build --release
RUN rm /rust/src/*.rs

# Download the map dataset, extraction will occur in-container to reduce image size
WORKDIR /maps
RUN wget https://storage.googleapis.com/super-metroid-map-rando/maps/session-2023-06-08T14:55:16.779895.pkl-bk24-subarea-balance-2.tgz \
    -O maps.tar.gz
RUN tar xfz maps.tar.gz --directory /maps && rm maps.tar.gz

# Now copy over the source code and build the real binary
COPY rust /rust
WORKDIR /rust
RUN cargo build --release --bin maprando-web

# Now restart with a slim base image and just copy over the binary and data needed at runtime.
FROM debian:buster-slim
RUN apt-get update && apt-get install -y \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*
COPY Mosaic /Mosaic
COPY compressed_data /compressed_data
COPY patches /patches
COPY gfx /gfx
COPY sm-json-data /sm-json-data
COPY MapRandoSprites /MapRandoSprites
COPY room_geometry.json /
COPY palette_smart_exports /palette_smart_exports
COPY visualizer /visualizer
# Both stages will run in parallel until the build stage is refernced,
# at which point this stage will wait for the `build` stage to complete, so delay these until last
COPY --from=build /maps /maps
COPY --from=build /rust/data /rust/data
COPY --from=build /rust/static /rust/static
# Since the bin is the most likely thing to have changed, copy it last to avoid invalidating the rest of the steps
COPY --from=build /rust/target/release/maprando-web /rust
WORKDIR /rust
ENTRYPOINT ["/rust/maprando-web"]
