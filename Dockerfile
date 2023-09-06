FROM rust:1.72.0 as build

# First get Cargo to download the crates.io index (which takes a long time)
RUN cargo new --bin rust
WORKDIR /rust
RUN cargo update

# Now use a dummy binary to build the project dependencies (allowing the results to be cached)
COPY rust/Cargo.lock /rust/Cargo.lock
COPY rust/Cargo.toml /rust/Cargo.toml
RUN cargo build --release
RUN rm /rust/src/*.rs

# Download and extract the map dataset
WORKDIR /maps
RUN wget https://storage.googleapis.com/super-metroid-map-rando/maps/session-2023-06-08T14:55:16.779895.pkl-bk24-subarea-balance-2.tgz
RUN mv session-2023-06-08T14:55:16.779895.pkl-bk24-subarea-balance-2.tgz maps.tar.gz && tar xfz maps.tar.gz

# Now copy over the source code and build the real binary
COPY rust /rust
WORKDIR /rust
RUN cargo build --release --bin maprando-web

# Now copy over data needed at runtime, that wasn't needed to build the binary.
COPY Mosaic /Mosaic
COPY compressed_data /compressed_data
COPY patches /patches
COPY gfx /gfx
COPY sm-json-data /sm-json-data
COPY MapRandoSprites /MapRandoSprites
COPY room_geometry.json /
COPY palette_smart_exports /palette_smart_exports
COPY visualizer /visualizer
WORKDIR /rust
CMD ["/rust/target/release/maprando-web"]
