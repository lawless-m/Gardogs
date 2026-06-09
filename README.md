# Gardogs

A reinforcement-learning tower defense in Rust: **dogs defend a garden** against
waves of cats (and later postmen and seagulls), and a DQN agent learns to play
**from the score alone** — no hand-coded strategy. See the design docs
(`00-overview.md` … `05-roadmap-phases.md`) for the full vision and plan.

## Workspace

| Crate            | Responsibility                                                      |
|------------------|--------------------------------------------------------------------|
| `gardogs-env`    | The game: rules, waves, economy, the `reset`/`step` API. Deterministic, headless. |
| `gardogs-agent`  | The learner: a hand-rolled MLP, replay buffer, and a Double-DQN training loop. |
| `gardogs-viz`    | The renderer/viewer. Placeholder until Phase 4.                    |
| `gardogs-cli`    | Driver binary (`gardogs`) for playing, training, and evaluating.   |

## Status: Phase 1 complete

The agent teaches itself to defend. `gardogs-env` implements the Phase-1 rules
(8×6 garden, one straight path, terriers vs cats, three starter waves, the
economy, win/lose), deterministic given a seed. `gardogs-agent` is a Double DQN —
MLP with manual backprop + Adam, experience replay, a target network, ε-greedy
exploration and gradient clipping — that learns, from the score alone, to place
dogs and survive the waves, reaching a 100% win rate from the random baseline.

## Try it

```sh
# Train a DQN on Phase 1 and save a checkpoint (a few minutes on CPU).
cargo run --release -p gardogs-cli -- train --seed 0 --episodes 600 --out phase1.ckpt

# Evaluate a saved checkpoint (greedy play: win rate / avg reward).
cargo run --release -p gardogs-cli -- eval --checkpoint phase1.ckpt --games 100

# Watch the random baseline lose, for contrast.
cargo run -p gardogs-cli -- play --seed 1 --agent-seed 1

# Run the tests (env rules + determinism; backprop, checkpointing, learning).
cargo test
```
