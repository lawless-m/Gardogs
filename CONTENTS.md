# Gardogs — Design Documents

A reinforcement-learning project: a tower-defense game where **dogs defend a
garden** against waves of cats, postmen, and seagulls — and an agent learns to
play it well **from nothing but the score**. No strategy is ever hand-coded; the
agent discovers good play by experience.

This bundle is the *design*, not the implementation. The intended workflow is:
these documents are handed to **Claude Code**, which writes the actual Rust.

## Where to start

Read in this order. Each builds on the last.

1. **`00-overview.md`** — the vision, the world, the philosophy, who does what,
   and which machine runs what. Read this first; it frames everything.
2. **`01-architecture.md`** — the Rust workspace layout, the crates, the tech
   stack (the neural-net library, the renderer), and GPU/backend notes for
   Debian and the two machines.
3. **`02-environment-api.md`** — the contract between the game and the learner:
   the `reset`/`step` interface, what an observation and an action are, and how
   the board is represented as numbers.
4. **`03-game-rules.md`** — the garden, the dogs, the enemies, the economy, the
   waves, win/lose — with concrete starting numbers to tune. This is the
   "rules and balance" core.
5. **`04-agent-and-training.md`** — how the agent actually learns: DQN, the
   replay buffer, the target network, reward design, exploration, hyperparameters.
6. **`05-roadmap-phases.md`** — the build order, Phase 0 → 5, with a clear
   "done" definition for each. **This is the document Claude Code should follow
   to decide what to build first.**

## The one-paragraph summary

A Rust workspace with a clean split: a deterministic **environment** crate (the
game and its rules, no graphics, no learning), an **agent** crate (the learner),
and a **viewer** crate (so a human can watch). The environment exposes a
Gym-style `reset`/`step` API so any learner can plug in unchanged. Build it in
phases: get a trivially simple version *running and learning* first (one dog,
one enemy, a flat grid), then add enemy variety, then richer spatial maps, then
the visuals that make it a joy to watch, and — as dessert — a second agent that
designs the attacking waves, so the system trains against itself.

## Status

This is a first complete design pass. Nothing here is fixed; it is a starting
point for discussion and tuning. The numbers in `03-game-rules.md` especially
are deliberate first guesses, meant to be balanced by playtesting and training.
