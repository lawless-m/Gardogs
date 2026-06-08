# 01 — Architecture

## The central split

Three concerns, kept apart, talking through narrow interfaces:

```
            reset / step                load checkpoint / query
  ┌────────────┐   ◄────────   ┌────────────┐   ◄────────   ┌────────────┐
  │ ENVIRONMENT │              │   AGENT    │              │   VIEWER    │
  │ (the game)  │   ────────►  │ (learner)  │   ────────►  │ (watch it)  │
  └────────────┘  obs/reward   └────────────┘  actions      └────────────┘
```

- The **environment** knows the rules and nothing about learning or graphics.
- The **agent** knows how to learn and nothing about how the game is drawn.
- The **viewer** knows how to draw and nothing about how the agent decided.

This separation is the most important architectural decision in the project.
It lets each part change independently, lets the environment be tested in
isolation, and means the same game works with any learner.

## Rust workspace layout

A single Cargo workspace with several crates:

| Crate           | Responsibility                                                                 | Depends on        |
|-----------------|--------------------------------------------------------------------------------|-------------------|
| `gardogs-env`   | The game: state, rules, waves, economy, the `reset`/`step` API. No graphics, no learning. Deterministic. | —                 |
| `gardogs-agent` | The learner: network, replay buffer, training loop, checkpoint save/load.      | `gardogs-env`     |
| `gardogs-viz`   | The renderer/viewer: draws the garden, plays back episodes, optionally live.   | `gardogs-env`     |
| `gardogs-cli`   | Thin binary(ies) to drive training runs, evaluation, and replay.               | the three above   |

`gardogs-env` is the heart. It must be:

- **Deterministic** — given a seed and a sequence of actions, the same game
  unfolds every time. Essential for reproducible training and for debugging.
  All randomness goes through one explicitly-seeded RNG.
- **Headless and fast** — no rendering, no I/O in the hot path. The training loop
  will run it millions of times.
- **Pure rules** — it answers "given this state and this action, what is the next
  state, the reward, and is the game over?" and nothing else.

## Tech stack

### Neural-network library: `burn`

`burn` is a mature, pure-Rust deep-learning framework. The reason it fits this
project particularly well is its **swappable backends**, which map cleanly onto
your machines and your Debian preference:

- **CPU (`burn-ndarray`)** — for development and the early phases. The networks
  are tiny; they train fine on CPU in seconds-to-minutes. Start here.
- **GPU via `burn-wgpu`** — runs on the 4070 with **no CUDA toolchain to
  install**, which avoids the usual driver/toolkit friction on Debian. This is
  the recommended GPU path at home.
- **GPU via CUDA / LibTorch** — maximum throughput on the work 3090 if/when a
  phase wants it. Optional, and only worth the setup cost later.

`candle` (from Hugging Face) is a reasonable alternative if `burn` ever proves
awkward; the architecture doesn't depend on the choice, since the network lives
entirely inside `gardogs-agent`.

### Renderer: start light

For Phase 4 (watching a trained agent play), a lightweight immediate-mode
renderer like **`macroquad`** is plenty — it's simple, cross-platform, and quick
to get a garden on screen. **`bevy`** is the heavier, more capable option if the
visual side grows ambitious; it's a natural fit for a game but more than the
early viewer needs. Decide at Phase 4, not now.

### Other crates (likely)

- `rand` (with a seedable, reproducible generator) for the environment RNG.
- `serde` for saving/loading game configs, checkpoints metadata, recorded episodes.
- a logging/metrics crate (e.g. `tracing`) so training progress is observable.

## Reproducibility and testing

Because `gardogs-env` is deterministic, it can be unit-tested hard: feed a seed
and a fixed action sequence, assert the resulting states. This is the main
defence against coordinate/geometry bugs — the kind of "which cell is the path
actually on" mistake that spatial reasoning is prone to. Keep the geometry small
and rigid early specifically so these tests are easy and exhaustive.

Training runs should log enough (seed, config hash, reward curve, checkpoints) to
be reproduced and compared. A 45-year veteran will want to diff two runs and
trust the difference is the change, not the weather.

## Build / run shape (for orientation, not prescription)

- `cargo run -p gardogs-cli -- train --config <file>` → runs a training session,
  writes checkpoints and a reward log.
- `cargo run -p gardogs-cli -- eval --checkpoint <file>` → plays N games with a
  trained agent, reports win rate / average score.
- `cargo run -p gardogs-cli -- watch --checkpoint <file>` → opens the viewer and
  plays a game live so you can see what it learned.

Exact CLI shape is Claude Code's call; this is just the intended set of verbs.
