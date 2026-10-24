#!/usr/bin/env python3
"""Download pool manifests, or complete selected pools, using map_pools.json."""

import argparse
from http.client import HTTPException
import json
from pathlib import Path
import sys
from urllib.parse import quote, urljoin, urlparse
from urllib.request import urlopen


RUST_DIR = Path(__file__).resolve().parents[1] / "rust"


def download(base_url, filename, destination):
    if destination.is_file():
        return
    url = urljoin(base_url.rstrip("/") + "/", quote(filename, safe=""))
    print(f"Downloading {url}", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with urlopen(url, timeout=30) as response, partial.open("wb") as output:
        output.write(response.read())
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pools", nargs="*", help="Pool identifiers from map_pools.json")
    parser.add_argument("--config", type=Path, default=RUST_DIR / "data/map_pools.json")
    parser.add_argument(
        "--manifests-only", action="store_true", help="Download only manifests (all pools by default)"
    )
    args = parser.parse_args()
    if not args.pools and not args.manifests_only:
        parser.error("Specify pools to download, or use --manifests-only")

    with args.config.open() as source:
        pools = json.load(source)["pools"]
    for name in args.pools:
        if name not in pools:
            parser.error(f"Unknown pool {name!r}; available pools: {', '.join(pools)}")

    for name in args.pools or pools:
        pool = pools[name]
        base_url = pool.get("base_url")
        if not base_url:
            print(f"{name}: local pool; no download needed")
            continue
        if urlparse(base_url).scheme not in ("http", "https"):
            raise ValueError(f"{name}: base_url must use HTTP or HTTPS")
        path = RUST_DIR.parent / "maps" / name
        manifest_path = path / "manifest.json"
        download(base_url, "manifest.json", manifest_path)
        with manifest_path.open() as source:
            manifest = json.load(source)
        if not manifest["files"]:
            raise ValueError(f"{name}: manifest contains no batch files")
        for filename in manifest["files"]:
            if not filename or Path(filename).name != filename or filename in (".", ".."):
                raise ValueError(f"{name}: batch filename must be a basename: {filename!r}")
        if not args.manifests_only:
            for filename in manifest["files"]:
                download(base_url, filename, path / filename)
        else:
            print(f"{name}: {'manifest' if args.manifests_only else 'pool'} ready")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, HTTPException) as error:
        sys.exit(str(error))
