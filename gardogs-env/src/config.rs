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

    /// The Phase 2 game from `03-game-rules.md`: the full bestiary and dog roster
    /// on the same 8×6 garden, with mixed, escalating waves — including waves
    /// where ground-only defence fails because seagulls come over the top.
    ///
    /// Enemy indices: 0 = cat, 1 = postman, 2 = seagull.
    /// Dog indices:   0 = terrier, 1 = mastiff, 2 = collie, 3 = german shepherd.
    ///
    /// Every number is a first guess to tune by training (the roadmap calls the
    /// wave scripts "a tuning exercise"); balance lives here, not in `game.rs`.
    pub fn phase2() -> Self {
        let cat = SpawnGroup {
            enemy: 0,
            count: 0,
            spacing_s: 0.0,
            start_s: 0.0,
        };
        let postman = SpawnGroup {
            enemy: 1,
            count: 0,
            spacing_s: 0.0,
            start_s: 0.0,
        };
        let seagull = SpawnGroup {
            enemy: 2,
            count: 0,
            spacing_s: 0.0,
            start_s: 0.0,
        };

        GameConfig {
            grid: GridConfig {
                width: 8,
                height: 6,
                path_row: 3,
            },
            tick_rate: 10.0,
            start_money: 120,
            start_lives: 12,
            breather_s: 4.0,
            dogs: vec![
                DogSpec {
                    name: "Terrier",
                    cost: 50,
                    damage: 5.0,
                    range: 1.5,
                    fire_rate: 1.0,
                    target: Some(Target::Ground),
                    slow_factor: None,
                },
                DogSpec {
                    name: "Mastiff",
                    cost: 120,
                    damage: 30.0,
                    // 03-game-rules.md lists 1.0, but buildable cells sit 1.0 cell
                    // off the path, so a dog covers ±sqrt(range^2 - 1) of it: range
                    // 1.0 grazes a single point and the anti-tank dog is useless.
                    // Bumped so the Mastiff can actually hold a chokepoint.
                    range: 1.5,
                    fire_rate: 0.4,
                    target: Some(Target::Ground),
                    slow_factor: None,
                },
                DogSpec {
                    name: "Border Collie",
                    cost: 80,
                    damage: 0.0,
                    range: 2.0,
                    fire_rate: 0.0,
                    target: None,
                    slow_factor: Some(0.5),
                },
                DogSpec {
                    name: "German Shepherd",
                    cost: 110,
                    damage: 12.0,
                    range: 1.5,
                    fire_rate: 0.8,
                    target: Some(Target::Both),
                    slow_factor: None,
                },
            ],
            enemies: vec![
                EnemySpec {
                    name: "Cat",
                    health: 20.0,
                    speed: 1.0,
                    movement: Movement::Ground,
                    kill_reward: 5.0,
                    kill_money: 10,
                    lives_cost: 1,
                },
                EnemySpec {
                    name: "Postman",
                    health: 100.0,
                    speed: 0.5,
                    movement: Movement::Ground,
                    kill_reward: 15.0,
                    kill_money: 25,
                    lives_cost: 2,
                },
                EnemySpec {
                    name: "Seagull",
                    health: 15.0,
                    speed: 1.5,
                    movement: Movement::Air,
                    kill_reward: 8.0,
                    kill_money: 12,
                    lives_cost: 1,
                },
            ],
            waves: vec![
                // 1: cats only — warm up and earn a little money.
                WaveSpec {
                    groups: vec![SpawnGroup {
                        count: 6,
                        spacing_s: 1.5,
                        ..cat
                    }],
                },
                // 2: cats plus seagulls over the top — a ground-only defence leaks.
                WaveSpec {
                    groups: vec![
                        SpawnGroup {
                            count: 6,
                            spacing_s: 1.5,
                            ..cat
                        },
                        SpawnGroup {
                            count: 3,
                            spacing_s: 2.5,
                            start_s: 3.0,
                            ..seagull
                        },
                    ],
                },
                // 3: cats and the first postmen — tanks that soak damage.
                WaveSpec {
                    groups: vec![
                        SpawnGroup {
                            count: 8,
                            spacing_s: 1.2,
                            ..cat
                        },
                        SpawnGroup {
                            count: 2,
                            spacing_s: 4.0,
                            start_s: 1.0,
                            ..postman
                        },
                    ],
                },
                // 4: postmen up the path while seagulls come over — needs both answers.
                WaveSpec {
                    groups: vec![
                        SpawnGroup {
                            count: 8,
                            spacing_s: 1.2,
                            ..cat
                        },
                        SpawnGroup {
                            count: 3,
                            spacing_s: 3.5,
                            start_s: 1.0,
                            ..postman
                        },
                        SpawnGroup {
                            count: 4,
                            spacing_s: 2.0,
                            start_s: 4.0,
                            ..seagull
                        },
                    ],
                },
                // 5: the finale — everything at once, faster.
                WaveSpec {
                    groups: vec![
                        SpawnGroup {
                            count: 10,
                            spacing_s: 1.0,
                            ..cat
                        },
                        SpawnGroup {
                            count: 4,
                            spacing_s: 3.0,
                            ..postman
                        },
                        SpawnGroup {
                            count: 5,
                            spacing_s: 1.8,
                            start_s: 3.0,
                            ..seagull
                        },
                    ],
                },
            ],
            max_tracked_enemies: 10,
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
