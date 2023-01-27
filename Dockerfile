FROM rust:1.67.0 as build

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

# Now restart with a slim base image and just copy over the binary and data needed at runtime.
FROM debian:buster-slim
COPY --from=build /maps /maps
COPY --from=build /patches /patches
COPY --from=build /gfx /gfx
COPY --from=build /sm-json-data /sm-json-data
COPY --from=build /room_geometry.json /
COPY --from=build /rust/data /rust/data
COPY --from=build /rust/static /rust/static
COPY --from=build /rust/target/release/maprando-web /rust
WORKDIR /rust
CMD ["/rust/maprando-web"]
