//! Experience replay: a ring buffer of transitions, sampled in random
//! minibatches to break the correlation between consecutive steps and to reuse
//! each experience many times (`04-agent-and-training.md`).

use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// One step of experience: `(state, action, reward, next_state, done)` plus the
/// legality mask at `next_state` (needed to mask the bootstrap target).
#[derive(Clone, Debug)]
pub struct Transition {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
    pub next_mask: Vec<bool>,
}

/// A fixed-capacity ring buffer. Once full, new transitions overwrite the oldest.
pub struct Replay {
    cap: usize,
    buf: Vec<Transition>,
    pos: usize,
}

impl Replay {
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0, "replay capacity must be positive");
        Replay {
            cap,
            buf: Vec::with_capacity(cap),
            pos: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn push(&mut self, t: Transition) {
        if self.buf.len() < self.cap {
            self.buf.push(t);
        } else {
            self.buf[self.pos] = t;
        }
        self.pos = (self.pos + 1) % self.cap;
    }

    /// Sample `batch` transitions uniformly at random (with replacement).
    pub fn sample(&self, batch: usize, rng: &mut ChaCha8Rng) -> Vec<&Transition> {
        (0..batch)
            .map(|_| &self.buf[rng.gen_range(0..self.buf.len())])
            .collect()
    }
}
