# Gardogs

A reinforcement-learning tower defense in Rust: **dogs defend a garden** against
waves of cats (and later postmen and seagulls), and a DQN agent learns to play
**from the score alone** — no hand-coded strategy. See the design docs
(`00-overview.md` … `05-roadmap-phases.md`) for the full vision and plan.

## Workspace

| Crate            | Responsibility                                                      |
|------------------|--------------------------------------------------------------------|
| `gardogs-env`    | The game: rules, waves, economy, the `reset`/`step` API. Deterministic, headless. |
| `gardogs-agent`  | The learner. Phase 0: a mask-driven random agent. DQN in Phase 1.  |
| `gardogs-viz`    | The renderer/viewer. Placeholder until Phase 4.                    |
| `gardogs-cli`    | Driver binary (`gardogs`) for running / evaluating games.          |

## Status: Phase 0 complete

The skeleton and the environment API are real. `gardogs-env` implements the
Phase-1 rules (8×6 garden, one straight path, terriers vs cats, three starter
waves, the economy, win/lose), it is deterministic given a seed, and a random
agent drives a full game end to end. No learning yet — that's Phase 1.

## Try it

```sh
# Run one full game with the random agent (it loses — there's no learning yet).
cargo run -p gardogs-cli -- play --seed 1 --agent-seed 1

# Run the tests (movement, firing, economy, win/lose, determinism).
cargo test
```
