//! `gardogs-env` — the Gardogs game as a deterministic, headless environment.
//!
//! This crate knows the *rules* and nothing about learning or graphics. It
//! exposes a Gym-style [`Env`] trait (`reset` / `step`) so any learner can plug
//! in unchanged. See the design docs `01-architecture.md` and `02-environment-api.md`.

mod config;
mod game;

pub use config::{
    DogSpec, EnemySpec, GameConfig, GridConfig, Movement, SpawnGroup, Target, WaveSpec,
};
pub use game::GardogsEnv;

/// The shape of an observation. Phase 1–2 use a flat vector; Phase 3+ will use
/// stacked 2-D planes for a convolutional network (`02-environment-api.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObsShape {
    Flat(usize),
    Planes {
        channels: usize,
        height: usize,
        width: usize,
    },
}

/// What the agent sees: the game state encoded as numbers, plus its shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub data: Vec<f32>,
    pub shape: ObsShape,
}

/// A decision the agent can take on a tick.
///
/// Actions are also addressed by a flat index in `0..action_count` (NOOP is 0),
/// which is what [`Env::action_mask`] is indexed by. Use
/// [`GardogsEnv::decode`] / [`GardogsEnv::place_index`] to convert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Do nothing this tick (let the game advance, save money).
    Noop,
    /// Place `dog` (a dog-spec index) on buildable cell `cell` (a buildable index).
    Place { dog: usize, cell: usize },
}

/// The result of advancing the game one tick.
#[derive(Clone, Debug, PartialEq)]
pub struct StepResult {
    pub obs: Observation,
    pub reward: f32,
    pub done: bool,
    pub info: Info,
}

/// Human-facing diagnostics. Never used for learning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Info {
    /// Cumulative reward so far this episode.
    pub score: f32,
    pub money: u32,
    pub lives: u32,
    /// Current wave index (0-based).
    pub wave: usize,
    pub tick: u64,
    pub enemies_active: usize,
    pub won: bool,
    /// Damage dogs dealt to enemies this tick. Diagnostic only — not part of the
    /// canonical reward; training may use it for optional shaping.
    pub damage_dealt: f32,
}

/// The contract between the game and the learner (`02-environment-api.md`).
pub trait Env {
    /// Start a fresh game with the given seed; returns the first observation.
    fn reset(&mut self, seed: u64) -> Observation;

    /// Apply one decision, advance the game one tick, and report what happened.
    fn step(&mut self, action: Action) -> StepResult;

    /// The shape of observations this environment produces.
    fn observation_shape(&self) -> ObsShape;

    /// How many distinct actions exist (including NOOP).
    fn action_count(&self) -> usize;

    /// Which actions are legal *right now* (affordable + legal placement),
    /// indexed by flat action index. The agent's source of truth.
    fn action_mask(&self) -> Vec<bool>;
}
