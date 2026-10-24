import json
import os
from pathlib import Path


def main():
    config_root = Path(".github/deploy")
    deploy_configs = {
        path.relative_to(config_root).with_suffix("").as_posix(): json.loads(path.read_text())
        for path in sorted(config_root.glob("*/*.json"))
    }
    if not deploy_configs:
        raise ValueError("No deployment configurations found in .github/deploy/*/*.json")
    metadata = {
        "digest": os.environ["IMAGE_DIGEST"],
        "commit": os.environ["GITHUB_SHA"],
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "deploy_configs": deploy_configs,
    }
    Path("image-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    repository_url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
    summary = (
        "### Published image\n\n"
        f"- Image: `docker.io/bkerby/map-rando@{metadata['digest']}`\n"
        f"- Commit: [{metadata['commit']}]({repository_url}/commit/{metadata['commit']})\n"
        f"- Build: [run {metadata['run_id']}, attempt {metadata['run_attempt']}]"
        f"({repository_url}/actions/runs/{metadata['run_id']}/attempts/{metadata['run_attempt']})\n"
    )
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary_file:
        summary_file.write(summary)


if __name__ == "__main__":
    main()
