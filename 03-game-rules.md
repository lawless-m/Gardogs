# 03 — Game Rules and Numbers

This is the rules-and-balance core. **Every number here is a deliberate first
guess**, meant to be tuned by playtesting and by watching what the agent does.
They're written down so Claude Code has something concrete to build, and so we
have a baseline to adjust from — not because they're sacred.

Units: distances are in **cells**, time in **seconds** (the simulator converts
seconds to ticks via the tick rate).

## The garden (Phase 1 geometry — deliberately boring)

- A grid, **8 wide × 6 tall**.
- One **path**: a straight lane along a single row, from the **gate** (left edge)
  to the **back door** (right edge). The thing being defended sits at the door.
- **Buildable cells**: every non-path cell. Dogs may only be placed on buildable,
  unoccupied cells.
- Ground enemies walk the path, cell to cell, from gate to door.

Keep this geometry rigid and minimal until the loop works. Interesting, winding
maps are a Phase 3 concern, steered by your taste (see `05-roadmap-phases.md`).

## What you're defending

- Start with **10 lives**.
- Each enemy that reaches the back door costs lives (per its type, below) and is
  removed.
- **0 lives → game over (loss).** Surviving all the waves → **win**.

## The economy

The classic tower-defense loop: spend money to place dogs, earn money by stopping
intruders, so the agent must learn to invest to grow.

- Start with **100 money**.
- Killing an enemy yields money (per type).
- Dogs cost money to place (per type). No selling in Phase 1 (add later if useful).

The interesting learned behaviour here is *tempo*: spend early to start earning,
but hold enough back for the tougher dog when a tank or a flyer arrives.

## Dogs (the defenders)

Introduced across phases. **Targeting** is the key mechanic: a dog hits `ground`,
`air`, or `both`. Seagulls can only be hit by `air`/`both` dogs — this is what
forces a differentiated defence.

| Dog            | Phase | Cost | Damage | Range (cells) | Fire rate (/s) | Targets | Notes                                   |
|----------------|-------|------|--------|---------------|----------------|---------|-----------------------------------------|
| Terrier        | 1     | 50   | 5      | 1.5           | 1.0            | ground  | Cheap, fast, low damage. The starter.   |
| Mastiff        | 2     | 120  | 30     | 1.0           | 0.4            | ground  | Expensive, slow, hard-hitting. A wall.  |
| Border Collie  | 2     | 80   | 0      | 2.0           | —              | —       | **Slows** enemies in range (e.g. ×0.5). Support, no damage. |
| German Shepherd| 2     | 110  | 12     | 1.5           | 0.8            | both    | The answer to seagulls; solid all-rounder. |

"Range" is radius from the dog's cell. A dog fires at the enemy in range it can
target, at its fire rate, dealing its damage per shot. (Targeting priority — e.g.
nearest, or furthest-along-path — is a small design knob; start with
"furthest along the path" so dogs protect the door.)

## Enemies (the attackers)

| Enemy    | Phase | Health | Speed (cells/s) | Movement        | Kill reward (score) | Kill money | Lives if it gets through |
|----------|-------|--------|-----------------|-----------------|---------------------|------------|--------------------------|
| Cat      | 1     | 20     | 1.0             | ground (path)   | +5                  | +10        | 1                        |
| Postman  | 2     | 100    | 0.5             | ground (path)   | +15                 | +25        | 2                        |
| Seagull  | 2     | 15     | 1.5             | **air** (straight over the fence to the door, ignores the path) | +8 | +12 | 1 |

The three form a rock-paper-scissors of *threat types*: the cat is the
fast-fragile swarm, the postman is the slow-tough wall, the seagull is the
airborne bypass. Each wants a different answer, which is the whole point.

## Waves

A wave is a scripted sequence of enemy spawns. Between waves there's a short
breather (time to build/upgrade). Difficulty escalates wave to wave.

Phase 1 starter waves (cats only):

| Wave | Spawns                    | Spacing |
|------|---------------------------|---------|
| 1    | 5 cats                    | every 2.0 s |
| 2    | 8 cats                    | every 1.5 s |
| 3    | 12 cats                   | every 1.2 s |

Phase 2 introduces mixed waves (cats + postmen + seagulls), e.g. a wave that
sends postmen up the path *while* seagulls come over the top, so a purely
ground defence fails. Exact wave scripts are a tuning exercise — the structure
is "escalating count, escalating mix, escalating speed".

## Win / lose, formally

- **Win**: survive every scripted wave with at least 1 life. (The episode ends
  positively.)
- **Lose**: lives reach 0. (The episode ends; large negative terminal signal —
  see `04-agent-and-training.md`.)

## Knobs we'll most want to tune

In rough order of how much they'll matter:

1. **Economy balance** — start money, costs, kill rewards. If dogs are too cheap
   the game is trivial; too dear and the agent can never recover.
2. **Wave pacing** — too brutal and nothing survives long enough to learn; too
   gentle and there's no pressure to play well.
3. **Range/fire-rate vs enemy speed** — decides whether a single dog can hold a
   lane, which shapes the whole tactical texture.
4. **Slow strength (Collie)** — how much support play is worth.

These are exactly the "numbers" worth iterating on together once it runs.
