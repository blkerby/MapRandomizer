# Experimental: random water/heat environments (WIP)

Branch: `random-hazzard`

## Summary

Adds optional balanced environment randomization:

- **Water:** flood N random dry rooms + dry N random vanilla water rooms
- **Heat:** heat N random dry rooms + cool N random vanilla heated rooms

Each axis includes ROM FX patching, pause-map tile updates, and a **partial** logic overlay. Solvability probing runs during seed generation (web and CLI).

## Not merge-ready for logic

Overlays mutate room JSON and patch some link requirements after `GameData` load, but they do **not** rebuild strats/links for:

- Full water physics (jump height, walljump, speed booster invalidation, etc.)
- Heat frame costs on movement between nodes
- Removing Gravity from dried vanilla water rooms

This is a prototype for discussion with core Map Rando developers, not production logic.

## What works today

- Bidirectional balance (flood ↔ dry, heat ↔ cool) with cross-axis room exclusion
- ROM FX via donor-room templates (`WaterFxPatcher`, `HeatFxPatcher`, `DryFxPatcher`)
- Pause-map water/heat/dry indicators
- Water flood overlay: door `physics: water` + Gravity AND on affected links
- Heat overlay: `roomEnvironments.heated` + `room_geometry.heated`
- Web UI toggles under Experimental; CLI flags `--randomize-water-environments` / `--randomize-heat-environments`
- `RandomizerContext::prepare_solvable` probes item placement before accepting an overlay

## Known limitations

- Logic overlay is coarse; dev feedback confirms a full rewrite is needed for correct routing
- Web generation requires a verified solvable overlay (no unverified fallback)
- When both water and heat are enabled, at most **one balanced pair per axis** is used
- No spoiler-log section listing affected rooms yet
- Escape start mode skips logic overlay (FX/map may still apply depending on settings)

## Try it

```sh
cd rust && cargo run -p maprando-web -- --seed-repository-url mem
```

Open http://localhost:8080, enable **Randomize water** and/or **Randomize heat** under Experimental.

CLI example:

```sh
cd rust && cargo run -p maprando-cli -- \
  --randomize-water-environments \
  --randomize-heat-environments \
  --map-layout Standard
```

## Files of interest

| Area | Path |
|------|------|
| Water assignment + overlay | `rust/maprando/src/water_environment.rs` |
| Heat assignment + overlay | `rust/maprando/src/heat_environment.rs` |
| Probe / pair caps | `rust/maprando/src/environment_logic.rs` |
| Prepare + probe loop | `rust/maprando/src/randomize.rs` (`RandomizerContext`) |
| ROM FX | `rust/maprando/src/patch/{water,heat,dry}_fx.rs` |
| Web generation loop | `rust/maprando-web/src/main.rs` |
| Web UI | `rust/maprando-web/templates/generate/experimental_environment.html` |

## Tests

```sh
cd rust && cargo test -p maprando water_environment heat_environment
```
