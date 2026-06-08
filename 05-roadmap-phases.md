# 05 — Roadmap (the build order)

This is the document to build *from*. Each phase has a clear "done" so progress
is unambiguous, and each phase produces something that works before the next adds
to it. **Resist building ahead** — the value of the early phases is that they
prove the loop cheaply.

---

## Phase 0 — Skeleton and API (no learning yet)

Get the bones standing and the contract real.

- The Rust workspace and the four crates (`01-architecture.md`).
- `gardogs-env` implementing `reset`/`step` for the Phase-1 rules
  (`03-game-rules.md`): the 8×6 garden, one straight path, terriers, cats,
  the three starter waves, the economy, win/lose.
- A trivial **random/scripted agent** that picks legal actions via the mask —
  just to drive the API and prove a full game runs end to end, headless.
- Deterministic given a seed; unit tests covering movement, firing, economy,
  and win/lose transitions.

**Done when:** a full game plays start to finish through `reset`/`step` with a
random agent, deterministically, and the tests pass. *No learning has happened —
that's expected.*

---

## Phase 1 — Dead-simple learning ("just get something running")

The first real "it taught itself" milestone, and the project's centre of gravity.

- `gardogs-agent`: the MLP, replay buffer, target network, ε-greedy, Double DQN,
  the training loop (`04-agent-and-training.md`).
- Compact-vector observation (`02-environment-api.md`).
- Train on the Phase-1 rules (terriers vs cats).

**Done when:** the reward curve clearly trends up and a trained agent reliably
survives the starter waves far better than the random baseline — i.e. it has
visibly learned to place dogs to defend the lane, with no strategy hand-coded.
This should train in well under an hour on home hardware (likely on CPU).

---

## Phase 2 — Enemy and dog variety (differentiated threats)

Make the tactics interesting.

- Add postmen and seagulls; add mastiff, collie, German shepherd
  (`03-game-rules.md`). Implement `ground`/`air`/`both` targeting and the
  collie's slow.
- Mixed, escalating waves — including waves where ground-only defence fails
  because seagulls come over the top.
- Retrain.

**Done when:** the agent demonstrably learns *differentiated* responses — e.g. it
fields `air`-capable dogs when seagulls appear and tougher dogs against postmen,
rather than spamming one dog type. This is the payoff: emergent, threat-specific
strategy from the score alone.

---

## Phase 3 — Spatial representation and richer maps

Let geometry get interesting, and let the agent reason about it properly.

- Switch the observation to stacked 2D planes; switch the network to the small
  CNN (`02`, `04`).
- Larger and/or **winding** maps, multiple paths, more buildable structure.
- **This is where your taste drives the map design** — interesting chokepoints,
  layout, feel. The agent's spatial competence now comes from convolutions, which
  is the right tool for it; the *look and interest* of the maps is a human call,
  with Claude Code implementing.
- Retrain on the richer maps.

**Done when:** the agent plays competently on a non-trivial, winding map using the
spatial input — handling geometry that the compact vector couldn't have
represented well.

---

## Phase 4 — Visuals (the part that matters to you)

Make it a joy to watch. This is a first-class goal, not an afterthought.

- `gardogs-viz`: render the garden — the dogs by breed, cats slinking in,
  postmen plodding, seagulls diving — and play back a trained agent live.
- Driven by **your visual direction**, leaning on real art/tile assets rather
  than invented-from-scratch geometry, with Claude Code doing the rendering
  (its strength).
- A "watch" mode: load a checkpoint, play a game on screen at human speed.

**Done when:** you can sit and watch the trained agent defend the garden, and it
looks the way *you* want it to.

---

## Phase 5 — Adversarial self-play (dessert)

The idea from the very start of the conversation, earned at the end.

- A **second** agent that *designs the attacking waves*, trained to beat the
  defending agent, while the defender trains to beat it. The system improves by
  playing against itself — no human strategy on either side.
- Best run on the work 3090, where the extra throughput pays off.

**Done when:** the two agents co-evolve — attacks get cleverer, defences adapt —
producing strategies neither we nor a fixed wave script would have written.
This is open-ended and exploratory by nature; "done" is really "interesting".

---

## A note on order and discipline

The temptation will be to jump to Phase 3/4 because they're the exciting,
visible parts. The reason not to: every later phase is far cheaper and less
buggy on top of a *working, tested* earlier phase. Phase 1 running — an agent
that visibly learns — is the milestone that de-risks everything after it. Get
there first.
