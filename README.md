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

After cloning the GitHub repository, download and extract required external data, which includes the randomized map pools and the Mosaic theming patches:

```sh
sh scripts/download_data.sh
```

Optionally, if you want to build and use the WebAssembly (for the boss calculator):

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

### Use local copy of videos

If you want to be able to have full functionality of the site while offline, including being able to browse the library of videos in the Logic pages, a local copy of the video library can be downloaded or synced as follows:

```sh
sh scripts/download_videos.sh
```

This will require at least 10 GB of disk space; running this multiple times will only download files that are new or have been updated. Then run the randomizer using the `--video-storage-path` option to point the randomizer to the local copy of the videos:

```sh
cd rust
cargo run --bin maprando-web -- --seed-repository-url mem --video-storage-path ../map-rando-videos
```

### Run the CLI using Cargo

As an alternative to using the web service, a CLI tool can also be used to generate a seed,  to get results with fewer steps. At the moment, the CLI tool has many randomization options hard-coded into it and is intended for development rather than general use.

After cloning the GitHub repository and downloading/extracting the maps (as above), run the CLI tool like this:

```sh
cd rust
cargo run --bin maprando-cli -- --map ../maps/v110c-wild/10000.json --input-rom YOUR-PATH-TO-VANILLA-ROM --output-rom OUTPUT-ROM-FILENAME
```
