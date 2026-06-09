//! `gardogs-agent` — the learner.
//!
//! - [`RandomAgent`] — a trivial mask-driven policy (drove the Phase 0 loop).
//! - [`Dqn`] — the Phase 1 Double-DQN learner (MLP, replay buffer, target
//!   network, ε-greedy), with a [`train`] loop and greedy [`evaluate`].
//!
//! See `04-agent-and-training.md`.

// The neural-net code uses explicit indexed loops over weight matrices (clearer
// than iterator chains for `W[o*in+i]` addressing), and integer modulo for the
// train/sync cadence (`is_multiple_of` needs a newer MSRV than we target).
#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

mod dqn;
mod nn;
mod replay;
mod train;

pub use dqn::{Dqn, DqnConfig};
pub use nn::{Adam, Mlp};
pub use replay::{Replay, Transition};
pub use train::{evaluate, train, Eval, TrainConfig, TrainReport};

use gardogs_env::Observation;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Anything that can choose a flat action index given an observation and the
/// environment's legality mask.
pub trait Policy {
    /// Choose a flat action index. The chosen index must be legal per `mask`.
    fn act(&mut self, obs: &Observation, mask: &[bool]) -> usize;
}

/// Picks a uniformly random legal action. Seeded for reproducibility.
pub struct RandomAgent {
    rng: ChaCha8Rng,
}

impl RandomAgent {
    pub fn new(seed: u64) -> Self {
        RandomAgent {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl Policy for RandomAgent {
    fn act(&mut self, _obs: &Observation, mask: &[bool]) -> usize {
        let legal: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &ok)| if ok { Some(i) } else { None })
            .collect();
        // NOOP is always legal, so `legal` is never empty.
        legal[self.rng.gen_range(0..legal.len())]
    }
}
