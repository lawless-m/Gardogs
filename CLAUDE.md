# Gardogs — notes for Claude Code

RL tower-defense in Rust. The design lives in `00-overview.md` … `05-roadmap-phases.md`;
**`05-roadmap-phases.md` is the build order — follow it and resist building ahead.**

## Architecture (see `01-architecture.md`)

Three concerns kept apart behind narrow interfaces:

- `gardogs-env` — the game. Knows the rules, nothing about learning or graphics.
  Must stay **deterministic** (one seeded RNG) and **headless/fast** (no I/O in
  the hot path). This is the heart; test it hard.
- `gardogs-agent` — the learner. Knows how to learn, nothing about rendering.
- `gardogs-viz` — the renderer. Knows how to draw, nothing about decisions.
- `gardogs-cli` — thin driver binary.

The `Env` trait (`reset`/`step`/`action_mask`, in `gardogs-env`) is the contract;
keep it narrow (`02-environment-api.md`).

## Conventions

- Game balance lives only in `GameConfig` (`config.rs`); `game.rs` reads it and
  hard-codes no numbers. Tune via config, not the simulator.
- Actions are addressed by a flat index (`0..action_count`, NOOP = 0); the action
  mask is indexed the same way and is the agent's source of truth. Illegal
  actions are treated as NOOP, never corrupting state.
- Keep determinism: given `(seed, action sequence)` the game must replay exactly.
  Add a test when you add a rule.

## Checks before committing

```sh
cargo test
cargo clippy --all-targets
cargo fmt --check
```

## Roadmap status

- **Phase 0 (done):** workspace, the four crates, `gardogs-env` Phase-1 rules,
  random agent, determinism + unit tests.
- **Phase 1 (done):** the DQN learner in `gardogs-agent` — a hand-rolled MLP with
  manual backprop + Adam (`nn.rs`), experience replay (`replay.rs`), a Double-DQN
  agent with a target net, ε-greedy and gradient clipping (`dqn.rs`), and a
  training loop with greedy eval and best-checkpoint tracking (`train.rs`). Trains
  terriers-vs-cats to a 100% win rate; the reward curve trends up from the random
  baseline. Drive it with `gardogs train` / `gardogs eval`.
- **Phase 2 (next):** enemy and dog variety — postmen and seagulls; mastiff,
  collie, German shepherd; `ground`/`air`/`both` targeting and the collie's slow
  (the env already models these); mixed, escalating waves; retrain until the agent
  fields differentiated, threat-specific defences. See `05-roadmap-phases.md`.

## Note on the learner

The net is hand-rolled in plain Rust (no `burn` yet): tiny, dependency-free, and
fully deterministic (every RNG seeded by us). DQN training is noisy and can
transiently forget, so `train()` keeps the best-evaluated network and restores it
at the end. If/when the spatial CNN phase (Phase 3) wants it, swapping in `burn`
is a localised change inside `gardogs-agent`.
