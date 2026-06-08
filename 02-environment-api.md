# 02 — Environment API

This is the contract between the game and the learner. It is deliberately narrow:
the agent only ever sees what flows through it, and the agent never touches the
game's internals.

## The interface, in plain terms

Two operations carry the whole conversation:

- **`reset`** — start a fresh game. Returns the first **observation** (the state
  the agent sees).
- **`step(action)`** — apply one decision, advance the game, and return:
  - the new **observation**,
  - a **reward** (a single number: how good that step turned out),
  - **done** (is the game over?),
  - some optional **info** (diagnostics for humans, never used for learning).

Plus two descriptors so the agent can size itself automatically:

- **observation shape** — how many numbers (and, later, what 2D layout) make up
  an observation.
- **action set** — how many distinct actions exist, and which are currently legal.

That is the entire API. Everything the agent knows about the world arrives
through `reset`/`step`; everything it does leaves through `action`.

## The trait (the one piece of interface worth pinning down)

This is the design artifact, to be *implemented* by Claude Code — sketched here
only because the signature communicates the contract faster than prose:

```rust
pub trait Env {
    fn reset(&mut self, seed: u64) -> Observation;

    fn step(&mut self, action: Action) -> StepResult;

    fn observation_shape(&self) -> ObsShape;
    fn action_count(&self) -> usize;

    /// Which actions are legal *right now* (affordable + legal placement).
    fn action_mask(&self) -> Vec<bool>;
}

pub struct StepResult {
    pub obs: Observation,
    pub reward: f32,
    pub done: bool,
    pub info: Info,
}
```

(Treat the exact types as provisional — they're the shape of the idea.)

## What an action is

Time in a tower defense flows continuously — enemies move every tick. The agent
does **not** have to act every tick, but the simplest clean design is to query it
each tick and let it choose to do nothing. So the action set is:

- **NOOP** — do nothing this tick (let the game advance, save money).
- **PLACE(dog_type, cell)** — for each dog type and each buildable cell.

For the dead-simple Phase 1 (one dog type, a small grid) this is a small,
flat set — e.g. `1 + (buildable_cells)` actions. As dog types and cells grow,
the set grows as `1 + (dog_types × buildable_cells)`.

### Action masking (important)

Most actions are illegal at any given moment — you can't afford a dog, or the
cell is occupied or on the path. The environment exposes an **action mask**
(`action_mask`) saying which actions are currently legal. The agent uses this to
ignore illegal choices (in practice, their value estimates are forced to minus
infinity before choosing). This dramatically stabilises and speeds up learning —
the agent never wastes experience discovering that it can't build on the path.

## What an observation is

The agent never sees a picture (until the spatial phase, and even then it sees
*planes of numbers*, not pixels). An observation is the game state encoded as
numbers. Two representations, introduced in different phases:

### Phase 1–2: a compact vector

A flat array of floats:

- **Board occupancy** — for each cell, what's there (empty / path / which dog).
  The grid is small enough to flatten into the vector.
- **Per-enemy summary** — position along the path, health, and type for the
  active enemies. To keep a fixed size, encode the nearest *N* enemies (and/or
  coarse per-cell enemy counts).
- **Scalars** — money, lives remaining, current wave, time until next wave.

This is small (tens to low-hundreds of numbers) and feeds a tiny fully-connected
network. It's what gets the project *running and learning* fast.

### Phase 3+: stacked 2D planes (for a convolutional network)

The same information laid out as the grid it really is — several aligned 2D
planes, one per feature:

- a plane marking the path,
- one plane per dog type (where that dog is placed),
- one plane per enemy type (where those enemies are),
- broadcast scalar planes for money / lives / wave.

A convolutional network reads these directly. This is the representation that
lets the map become large and winding without the input exploding, and it's
where the agent's genuine spatial competence comes from. (Crucially, this is the
*agent's* spatial reasoning via convolutions — not a language model's.)

The compact vector and the plane stack describe the same world; moving from one
to the other is a representation change, not a redesign. Designing the compact
grid as a flattened version of the eventual planes keeps that transition clean.

## Episodes, ticks, and timing

- An **episode** is one full game: a fixed number of waves, or until lives run out.
- A **tick** is one simulation step (enemies move, dogs fire, the clock advances).
  The agent is queried once per tick. Tick rate is a config value; it affects
  resolution, not the rules.
- `done` is true when the game ends (won all waves, or lost all lives).

## What the environment must guarantee

- **Determinism** given `(seed, action sequence)`.
- **Legality enforcement** — illegal actions are either masked out or treated as
  NOOP (never corrupt state). The mask is the agent's source of truth.
- **No hidden strategy** — the environment encodes *rules*, never tactics. It
  must not, for instance, auto-place a dog or nudge the agent. Every tactical
  decision is the agent's.
