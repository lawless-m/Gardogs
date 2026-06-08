# 00 — Overview

## The vision

A machine teaches *itself* to play a tower-defense game. It is handed the rules
(it can simulate "if I do this, what happens") but **no strategy whatsoever**.
It learns where to place defenders, when to save versus spend, and how to cover
different threats — purely from the win/lose/score signal, over many thousands
of self-played games.

The "aha" moment we are building toward: you watch it place defenders at random
and lose badly, and then — having been told nothing about tactics — it starts
guarding the gate, holding money back for a tougher dog, and learning that the
flying threat needs a different answer than the one on the ground. That emergent
competence, with no hand-coded heuristics, is the whole point.

## The world

The dogs defend a **back garden**. Everything that wants in is something a real
dog genuinely loses its mind over:

- **Cats** — fast, sneaky, fragile. Come over the fence / through the gate.
- **Postmen** — slow, plodding, tough. March up the path. The "tanks".
- **Seagulls** — *fly*. They ignore the fence and the ground path entirely,
  coming straight over the top. They need defenders that can deal with the air.

The defenders are **dogs**, placed around the garden, differing by breed:

- small, cheap, fast, low-damage (a terrier);
- big, expensive, slow, hard-hitting (a mastiff);
- a support breed that *slows* intruders (a collie);
- a breed that can reach the **air** as well as the ground.

What's being defended is something at the end of the path — the back door, say.
When an intruder gets through, you lose a life. Run out of lives and the game
is over.

This theme isn't decoration. It *generates* the mechanics: three genuinely
distinct threats (fast/fragile, slow/tough, airborne) force genuinely distinct
defensive answers, and that tension is the engine of any good tower defense —
and the thing that makes the agent's learning interesting to watch.

## Design philosophy

- **Rules known, strategy learned.** The agent gets a perfect simulator of the
  rules and learns tactics from experience. This is the sample-efficient,
  fast-converging road — not the pixels-only Atari road. Same "it taught itself"
  payoff, achievable on home hardware in well under an hour for the early phases.
- **A clean API boundary.** The game and the learner talk only through a narrow
  `reset`/`step` interface. Neither knows the other's internals, so we can swap
  either side freely. (This is the standard "Gym/Gymnasium" convention.)
- **Start trivially simple, grow in phases.** The first version is deliberately
  ugly and minimal, so it *runs and learns* fast. Complexity and polish are
  added deliberately, later, once the loop is proven. See `05-roadmap-phases.md`.
- **The look matters and is human-steered.** The visual presentation is a
  first-class goal (not an afterthought), but it is a deliberate later phase,
  driven by your taste and Claude Code's rendering strength — not invented from
  scratch by a model that is weak at spatial-visual design.

## Division of labour

A useful way to think about who is good at what:

- **You** — taste, direction, the "is this fun / does this look right" calls, and
  all final decisions. You set the visual intent; the rest serves it.
- **The agent (a neural net)** — the spatial learning. The part that handles the
  grid is a *convolutional* network, the tool built precisely for grid/spatial
  structure. The project's spatial competence comes from here, learned over
  thousands of games — not from a language model's (weak) spatial reasoning.
- **Claude (in this chat)** — the rules and the numbers: mechanics, balance, the
  reward design, the API and architecture. The systems-and-bookkeeping design.
- **Claude Code** — the implementation, and especially the *visual polish*, which
  is far more its strength than describing a look in prose is anyone's.

## Hardware targets

Everything in the early phases is tiny and will train comfortably — likely on
CPU alone, certainly on either GPU.

- **Home** — RTX 4070 (8 GB VRAM), 16 GB RAM, Debian (can also boot Win 11).
  More than enough for the compact-state phases; fine for the small-resolution
  spatial/CNN phase too. This is the primary development machine.
- **Work** — RTX 3090 (24 GB VRAM), dual-Xeon, 64 GB RAM. Overkill for the early
  work; useful for faster batches, larger experiments, and the adversarial
  self-play phase (Phase 5).

GPU is genuinely optional until the spatial phase. The point of Rust here is a
*fast simulator* — RL is sample-hungry and a tight environment buys you more
training per second, which matters more early than raw GPU throughput.

## A note on language

The whole project is in **Rust**. The RL ecosystem is mostly Python, but this
project builds its own learner rather than reaching for a packaged algorithm, so
the usual reason to prefer Python (its libraries) mostly doesn't apply — and the
two performance-critical parts (the simulator and the self-play loop) are exactly
Rust's home turf. You won't write any of it yourself; Claude Code does that.
