//! Static description of a game: the garden, the dogs, the enemies, the waves
//! and the economy. Every number here is a first-guess to be tuned (see the
//! design doc `03-game-rules.md`); the simulator (`game.rs`) reads these and
//! contains no hard-coded balance of its own.

/// What a dog can shoot at. Seagulls (air) can only be hit by `Air`/`Both`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Ground,
    Air,
    Both,
}

/// How an enemy travels. Ground enemies walk the path; air enemies fly straight
/// over it to the door (same 1-D progress axis, but only `Air`/`Both` dogs hit them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Movement {
    Ground,
    Air,
}

/// One breed of defender.
#[derive(Clone, Debug)]
pub struct DogSpec {
    pub name: &'static str,
    pub cost: u32,
    pub damage: f32,
    /// Radius, in cells, from the dog's own cell.
    pub range: f32,
    /// Shots per second. `0.0` means the dog never fires (e.g. a pure support breed).
    pub fire_rate: f32,
    /// What it can hit. `None` for a non-attacking support breed.
    pub target: Option<Target>,
    /// If set, enemies in range have their speed multiplied by this each tick
    /// (e.g. `0.5`). Independent of firing; this is the support/slow mechanic.
    pub slow_factor: Option<f32>,
}

/// One kind of intruder.
#[derive(Clone, Debug)]
pub struct EnemySpec {
    pub name: &'static str,
    pub health: f32,
    /// Cells per second along the path.
    pub speed: f32,
    pub movement: Movement,
    /// Reward fed to the agent when this enemy is killed (the "score").
    pub kill_reward: f32,
    /// Money added to the economy when this enemy is killed.
    pub kill_money: u32,
    /// Lives lost if this enemy reaches the door.
    pub lives_cost: u32,
}

/// A homogeneous burst of spawns within a wave: `count` of `enemy`, the first at
/// `start_s` seconds into the wave and one every `spacing_s` after.
#[derive(Clone, Debug)]
pub struct SpawnGroup {
    pub enemy: usize,
    pub count: u32,
    pub spacing_s: f32,
    pub start_s: f32,
}

/// A wave is one or more spawn groups (Phase 1 waves are a single group;
/// Phase 2 mixes groups, e.g. postmen on the ground while seagulls come over).
#[derive(Clone, Debug)]
pub struct WaveSpec {
    pub groups: Vec<SpawnGroup>,
}

/// The garden geometry.
#[derive(Clone, Debug)]
pub struct GridConfig {
    pub width: usize,
    pub height: usize,
    /// The single row the path runs along (Phase 1: one straight lane).
    pub path_row: usize,
}

/// The full, static description of a game.
#[derive(Clone, Debug)]
pub struct GameConfig {
    pub grid: GridConfig,
    /// Simulation ticks per second. Converts the seconds-based specs above into ticks.
    pub tick_rate: f32,
    pub start_money: u32,
    pub start_lives: u32,
    /// Breather between waves, in seconds (time to build).
    pub breather_s: f32,
    pub dogs: Vec<DogSpec>,
    pub enemies: Vec<EnemySpec>,
    pub waves: Vec<WaveSpec>,
    /// How many enemies the compact observation tracks individually (nearest N).
    pub max_tracked_enemies: usize,
}

impl GameConfig {
    /// The Phase 1 game from `03-game-rules.md`: an 8×6 garden with one straight
    /// path, terriers vs cats, three escalating starter waves.
    pub fn phase1() -> Self {
        GameConfig {
            grid: GridConfig {
                width: 8,
                height: 6,
                path_row: 3,
            },
            tick_rate: 10.0,
            start_money: 100,
            start_lives: 10,
            breather_s: 3.0,
            dogs: vec![DogSpec {
                name: "Terrier",
                cost: 50,
                damage: 5.0,
                range: 1.5,
                fire_rate: 1.0,
                target: Some(Target::Ground),
                slow_factor: None,
            }],
            enemies: vec![EnemySpec {
                name: "Cat",
                health: 20.0,
                speed: 1.0,
                movement: Movement::Ground,
                kill_reward: 5.0,
                kill_money: 10,
                lives_cost: 1,
            }],
            waves: vec![
                WaveSpec {
                    groups: vec![SpawnGroup {
                        enemy: 0,
                        count: 5,
                        spacing_s: 2.0,
                        start_s: 0.0,
                    }],
                },
                WaveSpec {
                    groups: vec![SpawnGroup {
                        enemy: 0,
                        count: 8,
                        spacing_s: 1.5,
                        start_s: 0.0,
                    }],
                },
                WaveSpec {
                    groups: vec![SpawnGroup {
                        enemy: 0,
                        count: 12,
                        spacing_s: 1.2,
                        start_s: 0.0,
                    }],
                },
            ],
            max_tracked_enemies: 8,
        }
    }

    /// A tiny, deterministic scenario used by the tests: a single short wave and a
    /// one-shot dog, so win/lose/firing outcomes are easy to assert. Two starting
    /// lives so an undefended run loses quickly.
    pub fn test_minimal() -> Self {
        GameConfig {
            grid: GridConfig {
                width: 8,
                height: 6,
                path_row: 3,
            },
            tick_rate: 10.0,
            start_money: 100,
            start_lives: 2,
            breather_s: 1.0,
            dogs: vec![DogSpec {
                name: "TestDog",
                cost: 50,
                damage: 50.0, // one-shots a cat
                range: 2.0,
                fire_rate: 2.0,
                target: Some(Target::Ground),
                slow_factor: None,
            }],
            enemies: vec![EnemySpec {
                name: "Cat",
                health: 20.0,
                speed: 1.0,
                movement: Movement::Ground,
                kill_reward: 5.0,
                kill_money: 10,
                lives_cost: 1,
            }],
            waves: vec![WaveSpec {
                groups: vec![SpawnGroup {
                    enemy: 0,
                    count: 3,
                    spacing_s: 2.0,
                    start_s: 0.0,
                }],
            }],
            max_tracked_enemies: 8,
        }
    }
}
