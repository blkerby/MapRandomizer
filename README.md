# Super Metroid Map Rando

This is the repository for the [Super Metroid Map Rando](https://maprando.com) project, which randomly rearranges how the Super Metroid rooms connect to each other, creating fresh randomized worlds for players to explore.

## Development

If you are interested in contributing, feel free to reach out on the [Discord](https://discord.gg/Gc99YV2ZcB). There are a few ways to run the randomizer for development:

### Run the web service using Docker

Install [Docker](https://docs.docker.com/get-docker/) if it is not already installed on your system. 

Clone the repository:

```sh
git clone --recurse-submodules https://github.com/blkerby/MapRandomizer
cd MapRandomizer
```

#### Docker Compose build and run

```sh
docker-compose up --build
```

#### Manual Docker build and run

Build the Docker image:

```sh
docker build . -t map-rando
```

Run the web service:

```sh
docker run -p 0.0.0.0:8080:8080 map-rando /rust/maprando-web --seed-repository-url mem
```

Open a browser and navigate to [localhost:8080](http://localhost:8080) and you should see your locally running copy of the [Map Rando website](https://maprando.com).

With the option "--seed-repository-url mem", any randomized seeds that you generate are stored in memory. You can also use "--seed-repository-url file:my-seeds" to store the seed data as local files under "my-seeds", where you could inspect the data.

### Run the web service using Cargo

Building and running locally using Cargo is generally faster than using Docker, as you can take advantage of incremental compilation.

Install the stable Rust toolchain (e.g. using [rustup](https://rustup.rs/)).

Clone the GitHub repository:

```sh
git clone --recurse-submodules https://github.com/blkerby/MapRandomizer
cd MapRandomizer
```

Download and extract the pool of randomized maps:

```sh
mkdir maps && cd maps
wget https://storage.googleapis.com/super-metroid-map-rando/maps/session-2023-06-08T14:55:16.779895.pkl-small-34-subarea-balance-2.tgz
mv session-2023-06-08T14:55:16.779895.pkl-small-34-subarea-balance-2.tgz maps.tar.gz && tar xfz maps.tar.gz
cd ..
```

Run the web service:

```sh
cd rust
cargo run --bin maprando-web -- --seed-repository-url mem
```

### Run the CLI using Cargo

As an alternative to using the web service, a CLI tool can also be used to generate a seed,  to get results with fewer steps. At the moment, the CLI tool has many randomization options hard-coded into it and is intended for development rather than general use.

After cloning the GitHub repository and downloading/extracting the maps (as above), run the CLI tool like this:

```sh
cd rust
cargo run --bin maprando-cli -- --map ../maps/session-2023-06-08T14:55:16.779895.pkl-small-34-subarea-balance-2/10005.json --item-placement-seed 1 --input-rom YOUR-PATH-TO-VANILLA-ROM --output-rom OUTPUT-ROM-FILENAME
```
