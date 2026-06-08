# 04 — The Agent and How It Learns

## The approach: Deep Q-Learning (DQN)

This is the family DeepMind used for the Atari work, and it fits a grid-based
game with a discrete set of actions well. In one sentence:

> The agent learns a function that, for the current state, estimates **how good
> each possible action is** (its expected eventual reward), and then it mostly
> picks the best one.

That function — the **Q-function** — is a neural network. It is never told which
action is good; it discovers the values by playing, being rewarded and punished,
and adjusting.

Why DQN rather than the rules-known, search-based (AlphaZero) approach: DQN needs
no forward search at decision time, is simple to implement and debug, and is the
fastest route to "running and learning". The search-based approach is a possible
later evolution, but it's not the place to start. (The adversarial idea in
Phase 5 is where self-play re-enters, separately.)

## The network

- **Phase 1–2 (compact vector input):** a small **multilayer perceptron** — the
  flat observation in, a couple of hidden layers (~128 units each), and one
  output per action giving that action's estimated value. Tiny; trains on CPU.
- **Phase 3+ (2D-plane input):** a small **convolutional network** reading the
  stacked planes, then a few dense layers to the per-action outputs. This is the
  part that genuinely reasons about space.

The output is always "a value for each action". Combined with the **action mask**
from the environment (`02-environment-api.md`), illegal actions are forced to
minus infinity before the agent chooses, so it only ever picks legal moves.

## The three pieces that make DQN actually work

These are the tricks that turned DQN from "doesn't converge" into the Atari
results. All three are simple and exactly the kind of tight loop Rust does well.

1. **Experience replay.** Every transition `(state, action, reward, next_state,
   done)` is stored in a large ring buffer. Training samples *random* minibatches
   from it, rather than learning from consecutive frames. This breaks the
   correlation between successive states and reuses each experience many times.
2. **Target network.** A second, frozen copy of the network produces the "what
   the value should have been" targets. It's updated to match the live network
   only periodically (or slowly/Polyak-averaged). Without this, the network
   chases its own moving estimates and diverges.
3. **ε-greedy exploration.** The agent acts randomly a fraction ε of the time and
   greedily otherwise, with ε starting high (~1.0, explore everything) and
   decaying toward a small floor (~0.05). This is how it discovers good play
   before it knows any.

**Double DQN** (use the live net to pick the best next action, the target net to
value it) is a cheap, well-proven accuracy improvement — worth including from the
start. **Dueling DQN** and **prioritised replay** are optional later refinements.

## Reward design (the signal that shapes everything)

The reward is the *only* thing telling the agent what "good" means, so it must
encode exactly the behaviour we want and nothing accidental. Starting design:

| Event                               | Reward         |
|-------------------------------------|----------------|
| Enemy killed                        | + its kill reward (see `03`) |
| Wave survived                       | +10            |
| Life lost (enemy reaches the door)  | −1 per life    |
| Game lost (lives hit 0)             | −20 terminal   |
| Game won (all waves survived)       | +50 terminal   |

Deliberately **no per-tick "survival" reward** at first — it tends to make the
agent loiter or game the clock. The kill/life/win signals are enough to learn
from, and `gamma` (below) propagates the delayed value backward so the agent
learns that a placement made now mattered for a wave cleared later. If learning
proves too sparse, a *small* shaping term (e.g. tiny reward for damage dealt) can
be added carefully — but only if needed, and watched for side-effects.

## Key hyperparameters (starting points)

| Parameter                  | Start value     | Note                                            |
|----------------------------|-----------------|-------------------------------------------------|
| Discount factor `gamma`    | 0.99            | High, because rewards are delayed.              |
| Replay buffer size         | 100,000         | Transitions.                                    |
| Minibatch size             | 64              |                                                 |
| Learning rate              | 1e-4            | Adam optimiser.                                 |
| Target net update          | every 1,000 steps | Or Polyak with tau ≈ 0.005.                   |
| ε schedule                 | 1.0 → 0.05      | Decay over the first ~50k–100k steps.           |
| Warmup before learning     | ~1,000 steps    | Fill the buffer a little before training starts.|

All provisional; these are sane DQN defaults to tune from.

## The training loop, in plain terms

Repeat for many games:

1. `reset` the environment.
2. Each tick: read the observation, apply the mask, choose an action (ε-greedy),
   `step` it, and store the resulting transition in the replay buffer.
3. Every few steps: sample a random minibatch from the buffer and nudge the
   network so its value estimates better match the observed rewards plus the
   (target-network) value of where it ended up.
4. Periodically sync the target network and decay ε.
5. Periodically: save a checkpoint, and evaluate (play some games with
   exploration off) to measure real progress.

## How we'll know it's working

- **Reward curve trending up** over training — the headline signal.
- **Win rate** (in greedy evaluation) climbing across waves survived.
- **Watchable behaviour** (Phase 4): it stops placing dogs at random, starts
  covering the gate, holds money for the right dog, and — the moment we're after —
  learns that seagulls need an `air`-capable dog, with no one having told it so.

## Reproducibility

Seed everything (environment and network init). Log seed, config, and reward
curve per run so two runs can be diffed and the difference trusted. This matters
for tuning the numbers in `03` sanely rather than chasing noise.
