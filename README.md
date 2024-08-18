# Super Metroid Map Rando

This is the repository for the [Super Metroid Map Rando](https://maprando.com) project, which randomly rearranges how the Super Metroid rooms connect to each other, creating fresh randomized worlds for players to explore.

## Development

If you are interested in contributing, feel free to reach out on the [Discord](https://discord.gg/Gc99YV2ZcB). There are a few ways to run the randomizer for development, described in detail below.

- Run the web service using Docker
- Run the web service using Cargo
- Run the CLI using Cargo

### Using Windows

If you are running on Windows, be sure to enable symlinks in Git before cloning the repository:

```sh
git config --global core.symlinks true
```

You likely also need to enable "Developer Mode" in Windows settings in order for Git to have permissions to create symlinks.

### Clone the repository

Clone the repository:

```sh
git clone --recurse-submodules https://github.com/blkerby/MapRandomizer
cd MapRandomizer
```

### Run the web service using Docker

Install [Docker](https://docs.docker.com/get-docker/) if it is not already installed on your system. 


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

After cloning the GitHub repository, download and extract the randomized map pools:

```sh
cd maps
wget https://storage.googleapis.com/super-metroid-map-rando/maps/v113-tame.tgz
tar xfz v113-tame.tgz && rm v113-tame.tgz
wget https://storage.googleapis.com/super-metroid-map-rando/maps/v110c-wild.tgz
tar xfz v110c-wild.tgz && rm v110c-wild.tgz
cd ..
```

Optionally, if you want to build and use the WebAssembly:

```sh
cd rust
cargo install wasm-pack
cd maprando-wasm
wasm-pack build --target="web" --release
cd ../..
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
cargo run --bin maprando-cli -- --map ../maps/v110c-wild/10000.json --input-rom YOUR-PATH-TO-VANILLA-ROM --output-rom OUTPUT-ROM-FILENAME
```
