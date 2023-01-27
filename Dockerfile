FROM rust:1.67.0

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
RUN wget https://storage.googleapis.com/super-metroid-map-rando/session-2022-06-03T17:19:29.727911.pkl-bk30-subarea-balance.tar.gz
RUN mv session-2022-06-03T17:19:29.727911.pkl-bk30-subarea-balance.tar.gz maps.tar.gz && tar xfz maps.tar.gz

# Now copy over everything else and build the real binary
COPY patches /patches
COPY gfx /gfx
COPY sm-json-data /sm-json-data
COPY room_geometry.json /
COPY rust /rust
WORKDIR /rust
RUN cargo build --release --bin maprando-web

CMD ["target/release/maprando-web"]
