//! `gardogs-agent` — the learner.
//!
//! Phase 0 ships only [`RandomAgent`], a trivial policy that picks a uniformly
//! random *legal* action via the environment's action mask. Its job is to drive
//! the `reset`/`step` API end to end and prove a full game runs. The DQN learner
//! (replay buffer, target network, ε-greedy) arrives in Phase 1.

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
