//! The Double-DQN learner: an online network, a periodically-synced target
//! network, ε-greedy action selection over the legality mask, and experience
//! replay. See `04-agent-and-training.md`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::nn::{Adam, Mlp};
use crate::replay::{Replay, Transition};

/// DQN hyperparameters. Defaults follow the design doc's starting points.
#[derive(Clone, Debug)]
pub struct DqnConfig {
    pub hidden: Vec<usize>,
    pub adam: Adam,
    pub gamma: f32,
    pub batch: usize,
    pub buffer_cap: usize,
    /// Sync the target net to the online net every this many optimiser steps.
    pub target_update_every: u64,
    pub eps_start: f32,
    pub eps_end: f32,
    /// ε decays linearly from `eps_start` to `eps_end` over this many env steps.
    pub eps_decay_steps: u64,
    /// Fill the buffer this much before learning starts.
    pub warmup: usize,
    /// Run one optimiser step every this many env steps.
    pub train_every: u64,
    /// Clip the global gradient L2 norm to this before each Adam step.
    pub grad_clip: Option<f32>,
}

impl Default for DqnConfig {
    fn default() -> Self {
        DqnConfig {
            hidden: vec![128, 128],
            adam: Adam::default(),
            gamma: 0.99,
            batch: 64,
            buffer_cap: 100_000,
            target_update_every: 1_000,
            eps_start: 1.0,
            eps_end: 0.05,
            eps_decay_steps: 50_000,
            warmup: 1_000,
            train_every: 4,
            grad_clip: Some(10.0),
        }
    }
}

/// A Double-DQN agent over a discrete, masked action set.
pub struct Dqn {
    online: Mlp,
    target: Mlp,
    replay: Replay,
    cfg: DqnConfig,
    rng: ChaCha8Rng,
    /// Environment steps observed (drives ε and the train/sync cadence).
    env_steps: u64,
    /// Optimiser steps taken (drives target sync).
    opt_steps: u64,
}

impl Dqn {
    pub fn new(obs_dim: usize, action_count: usize, cfg: DqnConfig, seed: u64) -> Self {
        let mut init_rng = ChaCha8Rng::seed_from_u64(seed);
        let mut sizes = Vec::with_capacity(cfg.hidden.len() + 2);
        sizes.push(obs_dim);
        sizes.extend_from_slice(&cfg.hidden);
        sizes.push(action_count);

        let online = Mlp::new(&sizes, &mut init_rng);
        let mut target = Mlp::new(&sizes, &mut init_rng);
        target.copy_weights_from(&online);

        Dqn {
            online,
            target,
            replay: Replay::new(cfg.buffer_cap),
            // A separate stream for sampling/exploration, seeded off the init seed.
            rng: ChaCha8Rng::seed_from_u64(seed ^ 0xD9C5_3A7B_1E2F_4456),
            cfg,
            env_steps: 0,
            opt_steps: 0,
        }
    }

    pub fn epsilon(&self) -> f32 {
        let frac = (self.env_steps as f32 / self.cfg.eps_decay_steps as f32).min(1.0);
        self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac
    }

    pub fn env_steps(&self) -> u64 {
        self.env_steps
    }

    pub fn replay_len(&self) -> usize {
        self.replay.len()
    }

    /// Q-values for an observation (online network). Diagnostic / inspection.
    pub fn q_values(&self, obs: &[f32]) -> Vec<f32> {
        self.online.q(obs)
    }

    /// ε-greedy action: explore a legal action with probability ε, else pick the
    /// highest-valued legal action. Uses (and advances) the agent's own RNG.
    pub fn act_explore(&mut self, obs: &[f32], mask: &[bool]) -> usize {
        if self.rng.gen::<f32>() < self.epsilon() {
            random_legal(mask, &mut self.rng)
        } else {
            argmax_masked(&self.online.q(obs), mask)
        }
    }

    /// Greedy action (no exploration). For evaluation and play.
    pub fn act_greedy(&self, obs: &[f32], mask: &[bool]) -> usize {
        argmax_masked(&self.online.q(obs), mask)
    }

    /// Store a transition and, once warmed up and on cadence, take one learning
    /// step. Returns the training loss when a step was taken. Counts one env step.
    pub fn observe(&mut self, t: Transition) -> Option<f32> {
        self.replay.push(t);
        self.env_steps += 1;

        if self.replay.len() < self.cfg.warmup || self.env_steps % self.cfg.train_every != 0 {
            return None;
        }
        Some(self.learn_step())
    }

    /// Sample a minibatch and apply one Double-DQN gradient update.
    fn learn_step(&mut self) -> f32 {
        let batch = self.cfg.batch.min(self.replay.len());
        let samples = self.replay.sample(batch, &mut self.rng);

        // Compute the bootstrap targets first (immutable borrows of both nets).
        let mut targets = Vec::with_capacity(batch);
        for t in &samples {
            let y = if t.done {
                t.reward
            } else {
                // Double DQN: online net picks the next action, target net values it.
                let next_online = self.online.q(&t.next_state);
                let a_star = argmax_masked(&next_online, &t.next_mask);
                let next_target = self.target.q(&t.next_state);
                t.reward + self.cfg.gamma * next_target[a_star]
            };
            targets.push(y);
        }

        // Accumulate gradients over the batch (Huber loss on the TD error of the
        // action actually taken; other outputs get zero gradient).
        let out_dim = self.online.output_dim();
        self.online.zero_grad();
        let mut total_loss = 0.0f32;
        for (t, &y) in samples.iter().zip(&targets) {
            let cache = self.online.forward_train(&t.state);
            let q_sa = cache.out[t.action];
            let e = q_sa - y;
            total_loss += huber(e);

            let mut grad_out = vec![0.0f32; out_dim];
            grad_out[t.action] = huber_grad(e);
            self.online.backward(&cache, &grad_out);
        }
        self.online
            .adam_step(self.cfg.adam, batch, self.cfg.grad_clip);

        self.opt_steps += 1;
        if self.opt_steps % self.cfg.target_update_every == 0 {
            self.target.copy_weights_from(&self.online);
        }

        total_loss / batch as f32
    }

    /// Serialise the online network to an in-memory buffer (for best-checkpoint
    /// tracking during a noisy training run).
    pub fn snapshot(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.online
            .save(&mut buf)
            .expect("in-memory save cannot fail");
        buf
    }

    /// Restore the online (and target) network from a [`Dqn::snapshot`].
    pub fn restore(&mut self, bytes: &[u8]) {
        self.online = Mlp::load(bytes).expect("restoring own snapshot cannot fail");
        self.target.copy_weights_from(&self.online);
    }

    // --- checkpointing ---------------------------------------------------------

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let f = std::fs::File::create(path)?;
        self.online.save(std::io::BufWriter::new(f))
    }

    /// Load an agent for inference/evaluation from a saved online network.
    pub fn load_for_eval(path: &str) -> std::io::Result<Dqn> {
        let f = std::fs::File::open(path)?;
        let online = Mlp::load(std::io::BufReader::new(f))?;
        let mut target = Mlp::new(
            &[online.input_dim(), online.output_dim()],
            &mut ChaCha8Rng::seed_from_u64(0),
        );
        target.copy_weights_from(&online);
        let cfg = DqnConfig::default();
        Ok(Dqn {
            online,
            target,
            replay: Replay::new(1),
            rng: ChaCha8Rng::seed_from_u64(0),
            cfg,
            env_steps: 0,
            opt_steps: 0,
        })
    }
}

/// Highest-valued legal action; falls back to NOOP (index 0) if none is legal.
fn argmax_masked(q: &[f32], mask: &[bool]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut found = false;
    for (i, (&qi, &ok)) in q.iter().zip(mask).enumerate() {
        if ok && qi > best_v {
            best_v = qi;
            best = i;
            found = true;
        }
    }
    if found {
        best
    } else {
        0
    }
}

fn random_legal(mask: &[bool], rng: &mut ChaCha8Rng) -> usize {
    let legal: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &ok)| if ok { Some(i) } else { None })
        .collect();
    legal[rng.gen_range(0..legal.len())]
}

/// Huber (smooth-L1) loss with delta = 1, the standard DQN choice.
fn huber(e: f32) -> f32 {
    if e.abs() <= 1.0 {
        0.5 * e * e
    } else {
        e.abs() - 0.5
    }
}

/// d(huber)/d(e), clipped to [-1, 1].
fn huber_grad(e: f32) -> f32 {
    e.clamp(-1.0, 1.0)
}
