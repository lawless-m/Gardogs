//! The deterministic simulator: state, the per-tick rules, and the [`Env`] impl.
//!
//! Given a seed and a sequence of actions, the same game unfolds every time.
//! There is no rendering and no learning here — only "given this state and this
//! action, what is the next state, the reward, and is it over?".

use std::rc::Rc;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::config::{GameConfig, Movement, Target};
use crate::{Action, Env, Info, ObsShape, Observation, StepResult};

/// A placed defender.
#[derive(Clone, Debug)]
struct Dog {
    /// Index into `cfg.dogs`.
    spec: usize,
    cell: (usize, usize),
    /// Ticks until it can fire again.
    cooldown: u32,
}

/// An active intruder.
#[derive(Clone, Debug)]
struct Enemy {
    /// Index into `cfg.enemies`.
    spec: usize,
    /// Distance travelled along the path, in cells (0 = gate). Reaches the door
    /// at `progress >= grid.width`.
    progress: f32,
    health: f32,
}

/// Where we are in the wave schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Spawning the current wave (and waiting for it to be cleared).
    Spawning,
    /// Breather between waves; advance to the next wave when it elapses.
    Breather { ticks_left: u32 },
}

/// The Gardogs environment. Construct with [`GardogsEnv::new`], then drive it
/// through the [`Env`] trait (`reset` / `step`).
pub struct GardogsEnv {
    cfg: Rc<GameConfig>,
    /// Buildable cells in a fixed order; `Action::Place.cell` indexes this.
    buildable: Vec<(usize, usize)>,
    /// Per grid-cell: is a dog standing here? (Row-major, `y * width + x`.)
    occupied: Vec<bool>,

    dogs: Vec<Dog>,
    enemies: Vec<Enemy>,
    money: u32,
    lives: u32,

    wave_idx: usize,
    /// Ticks since the current wave began spawning.
    wave_clock: u32,
    /// How many of each group in the current wave have spawned. Sized to the
    /// largest wave; only indices `< current_wave.groups.len()` are meaningful.
    spawned: Vec<u32>,
    phase: Phase,

    done: bool,
    won: bool,
    tick: u64,
    cumulative: f32,
    /// Total damage dogs dealt to enemies in the most recent tick. A diagnostic
    /// (it is not part of the canonical reward); training may use it for optional
    /// shaping. See `04-agent-and-training.md`.
    last_damage_dealt: f32,

    /// Seeded per `reset`. Reserved for future stochastic rules; the Phase 1
    /// rules are fully deterministic without it.
    #[allow(dead_code)]
    rng: ChaCha8Rng,
}

/// Can a dog with `target` hit an enemy with this `movement`?
fn target_hits(target: Target, movement: Movement) -> bool {
    match target {
        Target::Both => true,
        Target::Ground => movement == Movement::Ground,
        Target::Air => movement == Movement::Air,
    }
}

impl GardogsEnv {
    /// Build an environment for `cfg`. The game starts already reset (seed 0).
    pub fn new(cfg: GameConfig) -> Self {
        let cfg = Rc::new(cfg);
        let (w, h, pr) = (cfg.grid.width, cfg.grid.height, cfg.grid.path_row);

        let buildable: Vec<(usize, usize)> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(_, y)| y != pr)
            .collect();
        let max_groups = cfg
            .waves
            .iter()
            .map(|wv| wv.groups.len())
            .max()
            .unwrap_or(0);

        let mut env = GardogsEnv {
            cfg,
            buildable,
            occupied: vec![false; w * h],
            dogs: Vec::new(),
            enemies: Vec::new(),
            money: 0,
            lives: 0,
            wave_idx: 0,
            wave_clock: 0,
            spawned: vec![0; max_groups],
            phase: Phase::Spawning,
            done: false,
            won: false,
            tick: 0,
            cumulative: 0.0,
            last_damage_dealt: 0.0,
            rng: ChaCha8Rng::seed_from_u64(0),
        };
        env.reset(0);
        env
    }

    // --- helpers used by callers and tests -------------------------------------

    /// Decode a flat action index into an [`Action`].
    pub fn decode(&self, idx: usize) -> Action {
        if idx == 0 {
            Action::Noop
        } else {
            let i = idx - 1;
            let b = self.buildable.len();
            Action::Place {
                dog: i / b,
                cell: i % b,
            }
        }
    }

    /// The flat action index that places `dog` on grid cell `cell`, if that cell
    /// is buildable. Convenience for tests and scripted agents.
    pub fn place_index(&self, dog: usize, cell: (usize, usize)) -> Option<usize> {
        let c = self.buildable.iter().position(|&b| b == cell)?;
        Some(1 + dog * self.buildable.len() + c)
    }

    pub fn money(&self) -> u32 {
        self.money
    }
    pub fn lives(&self) -> u32 {
        self.lives
    }
    pub fn won(&self) -> bool {
        self.won
    }
    pub fn is_done(&self) -> bool {
        self.done
    }
    pub fn tick(&self) -> u64 {
        self.tick
    }
    pub fn enemies_active(&self) -> usize {
        self.enemies.len()
    }

    pub fn info(&self) -> Info {
        Info {
            score: self.cumulative,
            money: self.money,
            lives: self.lives,
            wave: self.wave_idx,
            tick: self.tick,
            enemies_active: self.enemies.len(),
            won: self.won,
            damage_dealt: self.last_damage_dealt,
        }
    }

    // --- the simulation core ---------------------------------------------------

    /// Advance the world one tick (spawn, fire, move, resolve waves) and return
    /// the reward accrued this tick.
    fn tick_world(&mut self) -> f32 {
        let cfg = self.cfg.clone();
        let width = cfg.grid.width as f32;
        let path_row = cfg.grid.path_row as f32;
        let mut reward = 0.0f32;
        self.last_damage_dealt = 0.0;

        // 1. Spawn any enemies due this tick.
        if let Phase::Spawning = self.phase {
            let wave = &cfg.waves[self.wave_idx];
            for g in 0..wave.groups.len() {
                let grp = &wave.groups[g];
                while self.spawned[g] < grp.count {
                    let t = ((grp.start_s + self.spawned[g] as f32 * grp.spacing_s) * cfg.tick_rate)
                        .round() as u32;
                    if t <= self.wave_clock {
                        let espec = &cfg.enemies[grp.enemy];
                        self.enemies.push(Enemy {
                            spec: grp.enemy,
                            progress: 0.0,
                            health: espec.health,
                        });
                        self.spawned[g] += 1;
                    } else {
                        break;
                    }
                }
            }
        }

        // 2. Dogs fire. Each attacking dog hits the enemy furthest along the path
        //    that it can target and that is in range.
        for di in 0..self.dogs.len() {
            let spec_idx = self.dogs[di].spec;
            let dspec = &cfg.dogs[spec_idx];
            let (target, fire_rate, damage, range) = match dspec.target {
                Some(t) if dspec.fire_rate > 0.0 => (t, dspec.fire_rate, dspec.damage, dspec.range),
                _ => continue, // support / non-firing breed
            };

            if self.dogs[di].cooldown > 0 {
                self.dogs[di].cooldown -= 1;
            }
            if self.dogs[di].cooldown != 0 {
                continue;
            }

            let (dx, dy) = self.dogs[di].cell;
            let (dxf, dyf) = (dx as f32, dy as f32);
            let mut best: Option<usize> = None;
            let mut best_progress = f32::NEG_INFINITY;
            for ei in 0..self.enemies.len() {
                let e = &self.enemies[ei];
                if !target_hits(target, cfg.enemies[e.spec].movement) {
                    continue;
                }
                let (ddx, ddy) = (e.progress - dxf, path_row - dyf);
                if (ddx * ddx + ddy * ddy).sqrt() <= range && e.progress > best_progress {
                    best_progress = e.progress;
                    best = Some(ei);
                }
            }

            if let Some(ei) = best {
                // Count damage actually dealt (no overkill) for optional shaping.
                self.last_damage_dealt += damage.min(self.enemies[ei].health.max(0.0));
                self.enemies[ei].health -= damage;
                if self.enemies[ei].health <= 0.0 {
                    let espec = &cfg.enemies[self.enemies[ei].spec];
                    reward += espec.kill_reward;
                    self.money += espec.kill_money;
                    self.enemies.swap_remove(ei);
                }
                let period = ((cfg.tick_rate / fire_rate).round() as u32).max(1);
                self.dogs[di].cooldown = period;
            }
        }

        // 3. Move enemies; any that reach the door leak (cost lives, are removed).
        let mut ei = 0;
        while ei < self.enemies.len() {
            let espec = &cfg.enemies[self.enemies[ei].spec];
            let pos = (self.enemies[ei].progress, path_row);

            // Strongest slow from any support dog in range (slows don't stack).
            let mut slow = 1.0f32;
            for d in &self.dogs {
                let dspec = &cfg.dogs[d.spec];
                if let Some(sf) = dspec.slow_factor {
                    let (ddx, ddy) = (pos.0 - d.cell.0 as f32, pos.1 - d.cell.1 as f32);
                    if (ddx * ddx + ddy * ddy).sqrt() <= dspec.range {
                        slow = slow.min(sf);
                    }
                }
            }

            self.enemies[ei].progress += (espec.speed / cfg.tick_rate) * slow;
            if self.enemies[ei].progress >= width {
                let lost = espec.lives_cost.min(self.lives);
                self.lives -= lost;
                reward -= lost as f32; // -1 per life lost
                self.enemies.swap_remove(ei); // last element moved into `ei`
            } else {
                ei += 1;
            }
        }

        // 4. Loss check (terminal).
        if self.lives == 0 {
            self.done = true;
            self.won = false;
            reward -= 20.0;
            return reward;
        }

        // 5. Advance the wave schedule.
        match self.phase {
            Phase::Spawning => {
                self.wave_clock += 1;
                let wave = &cfg.waves[self.wave_idx];
                let all_spawned =
                    (0..wave.groups.len()).all(|g| self.spawned[g] >= wave.groups[g].count);
                if all_spawned && self.enemies.is_empty() {
                    reward += 10.0; // wave survived
                    if self.wave_idx + 1 >= cfg.waves.len() {
                        self.done = true;
                        self.won = true;
                        reward += 50.0; // game won
                    } else {
                        let bt = ((cfg.breather_s * cfg.tick_rate).round() as u32).max(1);
                        self.phase = Phase::Breather { ticks_left: bt };
                    }
                }
            }
            Phase::Breather { ticks_left } => {
                let left = ticks_left - 1;
                if left == 0 {
                    self.wave_idx += 1;
                    self.wave_clock = 0;
                    for s in self.spawned.iter_mut() {
                        *s = 0;
                    }
                    self.phase = Phase::Spawning;
                } else {
                    self.phase = Phase::Breather { ticks_left: left };
                }
            }
        }

        reward
    }

    // --- observation -----------------------------------------------------------

    fn obs_len(&self) -> usize {
        let cfg = &self.cfg;
        let cells = cfg.grid.width * cfg.grid.height;
        cells                                   // path plane
            + cells * cfg.dogs.len()            // one presence plane per dog type
            + cfg.max_tracked_enemies * (2 + cfg.enemies.len()) // nearest-N enemies
            + 4 // money, lives, wave, breather
    }

    /// Build the compact-vector observation (`02-environment-api.md`, Phase 1–2).
    /// Laid out as flattened planes so the Phase 3 transition to 2-D is clean.
    fn build_obs(&self) -> Observation {
        let cfg = &self.cfg;
        let (w, h) = (cfg.grid.width, cfg.grid.height);
        let cells = w * h;
        let mut data = Vec::with_capacity(self.obs_len());

        // Path plane.
        for y in 0..h {
            let v = if y == cfg.grid.path_row { 1.0 } else { 0.0 };
            data.resize(data.len() + w, v);
        }

        // One presence plane per dog type.
        for d in 0..cfg.dogs.len() {
            let mut plane = vec![0.0f32; cells];
            for dog in &self.dogs {
                if dog.spec == d {
                    plane[dog.cell.1 * w + dog.cell.0] = 1.0;
                }
            }
            data.extend_from_slice(&plane);
        }

        // Nearest N enemies, furthest-along-the-path first (most urgent).
        let mut order: Vec<usize> = (0..self.enemies.len()).collect();
        order.sort_by(|&a, &b| {
            self.enemies[b]
                .progress
                .partial_cmp(&self.enemies[a].progress)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let etypes = cfg.enemies.len();
        for k in 0..cfg.max_tracked_enemies {
            if let Some(&idx) = order.get(k) {
                let e = &self.enemies[idx];
                let espec = &cfg.enemies[e.spec];
                data.push(e.progress / w as f32);
                data.push((e.health / espec.health).clamp(0.0, 1.0));
                for t in 0..etypes {
                    data.push(if t == e.spec { 1.0 } else { 0.0 });
                }
            } else {
                // Empty slot: progress, health, and the type one-hot, all zero.
                data.resize(data.len() + 2 + etypes, 0.0);
            }
        }

        // Scalars.
        data.push(self.money as f32 / 100.0);
        data.push(self.lives as f32 / cfg.start_lives.max(1) as f32);
        data.push(self.wave_idx as f32 / cfg.waves.len().max(1) as f32);
        let breather_norm = match self.phase {
            Phase::Breather { ticks_left } => {
                ticks_left as f32 / (cfg.breather_s * cfg.tick_rate).max(1.0)
            }
            Phase::Spawning => 0.0,
        };
        data.push(breather_norm);

        debug_assert_eq!(data.len(), self.obs_len());
        Observation {
            data,
            shape: ObsShape::Flat(self.obs_len()),
        }
    }
}

impl Env for GardogsEnv {
    fn reset(&mut self, seed: u64) -> Observation {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
        self.dogs.clear();
        for o in self.occupied.iter_mut() {
            *o = false;
        }
        self.enemies.clear();
        self.money = self.cfg.start_money;
        self.lives = self.cfg.start_lives;
        self.wave_idx = 0;
        self.wave_clock = 0;
        for s in self.spawned.iter_mut() {
            *s = 0;
        }
        self.phase = Phase::Spawning;
        self.done = false;
        self.won = false;
        self.tick = 0;
        self.cumulative = 0.0;
        self.last_damage_dealt = 0.0;
        self.build_obs()
    }

    fn step(&mut self, action: Action) -> StepResult {
        if self.done {
            // Episode already over: report the terminal state, change nothing.
            return StepResult {
                obs: self.build_obs(),
                reward: 0.0,
                done: true,
                info: self.info(),
            };
        }

        // Apply the action. Illegal placements are treated as NOOP (never corrupt
        // state) — the action mask is the agent's source of truth.
        if let Action::Place { dog, cell } = action {
            if dog < self.cfg.dogs.len() && cell < self.buildable.len() {
                let (x, y) = self.buildable[cell];
                let gi = y * self.cfg.grid.width + x;
                let cost = self.cfg.dogs[dog].cost;
                if self.money >= cost && !self.occupied[gi] {
                    self.money -= cost;
                    self.occupied[gi] = true;
                    self.dogs.push(Dog {
                        spec: dog,
                        cell: (x, y),
                        cooldown: 0,
                    });
                }
            }
        }

        let reward = self.tick_world();
        self.tick += 1;
        self.cumulative += reward;

        StepResult {
            obs: self.build_obs(),
            reward,
            done: self.done,
            info: self.info(),
        }
    }

    fn observation_shape(&self) -> ObsShape {
        ObsShape::Flat(self.obs_len())
    }

    fn action_count(&self) -> usize {
        1 + self.cfg.dogs.len() * self.buildable.len()
    }

    fn action_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; self.action_count()];
        mask[0] = true; // NOOP always legal
        if self.done {
            return mask; // nothing else is meaningful once the game is over
        }
        let b = self.buildable.len();
        for d in 0..self.cfg.dogs.len() {
            let affordable = self.money >= self.cfg.dogs[d].cost;
            for c in 0..b {
                let (x, y) = self.buildable[c];
                let gi = y * self.cfg.grid.width + x;
                mask[1 + d * b + c] = affordable && !self.occupied[gi];
            }
        }
        mask
    }
}
