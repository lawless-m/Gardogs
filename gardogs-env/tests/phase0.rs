//! Phase 0 acceptance tests: a full game runs through `reset`/`step`,
//! deterministically, with movement, firing, economy and win/lose all covered.

use gardogs_env::{Action, Env, GameConfig, GardogsEnv};

/// Run a game under a deterministic "highest legal action" policy, returning the
/// per-step reward trajectory plus the terminal state. Used to assert determinism.
fn run_greedy_highest(cfg: GameConfig, seed: u64) -> (Vec<f32>, u32, u32, bool, u64) {
    let mut env = GardogsEnv::new(cfg);
    env.reset(seed);
    let mut rewards = Vec::new();
    loop {
        let mask = env.action_mask();
        // Highest-index legal action: places a dog whenever one is affordable.
        let idx = (0..mask.len()).rev().find(|&i| mask[i]).unwrap();
        let action = env.decode(idx);
        let sr = env.step(action);
        rewards.push(sr.reward);
        if sr.done {
            break;
        }
        assert!(rewards.len() < 100_000, "game failed to terminate");
    }
    (rewards, env.lives(), env.money(), env.won(), env.tick())
}

/// Step the game with NOOP until it ends, returning the terminal env.
fn run_noop(mut env: GardogsEnv) -> GardogsEnv {
    loop {
        let sr = env.step(Action::Noop);
        if sr.done {
            break;
        }
        assert!(env.tick() < 100_000, "game failed to terminate");
    }
    env
}

#[test]
fn observation_and_action_shapes() {
    let env = GardogsEnv::new(GameConfig::phase1());
    // 8x6 grid, 1 dog type, 1 enemy type, N=8 tracked enemies:
    //   path(48) + dogs(48) + enemies(8 * (2+1)) + scalars(4) = 124
    use gardogs_env::ObsShape;
    assert_eq!(env.observation_shape(), ObsShape::Flat(124));
    // NOOP + 1 dog type * (48 - 8 path) buildable cells = 1 + 40
    assert_eq!(env.action_count(), 41);

    let obs = GardogsEnv::new(GameConfig::phase1()).reset(0);
    assert_eq!(obs.data.len(), 124);
}

#[test]
fn determinism_same_seed_same_trajectory() {
    let a = run_greedy_highest(GameConfig::phase1(), 7);
    let b = run_greedy_highest(GameConfig::phase1(), 7);
    assert_eq!(
        a, b,
        "same seed + same policy must reproduce the game exactly"
    );
}

#[test]
fn movement_undefended_loses() {
    // No dogs, ever: cats march to the door, lives drain, the game is lost.
    let env = GardogsEnv::new(GameConfig::test_minimal());
    let lives_start = env.lives();
    assert_eq!(lives_start, 2);

    let env = run_noop(env);
    assert!(env.is_done());
    assert!(!env.won());
    assert_eq!(env.lives(), 0, "undefended run should reach 0 lives");
}

#[test]
fn firing_economy_and_win() {
    // One strong dog beside the gate kills every cat -> survive the wave -> win.
    let mut env = GardogsEnv::new(GameConfig::test_minimal());
    env.reset(0);

    // Place the dog at (1, 2): adjacent to the path row (3), within its range.
    let place = env.place_index(0, (1, 2)).expect("(1,2) must be buildable");
    assert!(
        env.action_mask()[place],
        "placement should be legal with full money"
    );
    let sr = env.step(env.decode(place));
    assert!(!sr.done);
    // This same tick also advances the world: the first cat spawns at the gate
    // and the dog one-shots it. So 100 - 50 (dog) + 10 (kill) = 60.
    assert_eq!(env.money(), 60);
    assert_eq!(sr.reward, 5.0, "reward for the kill this tick");

    let env = run_noop(env);
    assert!(env.is_done());
    assert!(env.won(), "a one-shot dog at the gate should win");
    assert_eq!(env.lives(), 2, "no cat should have leaked");
    // Economy: start 100 - 50 (dog) + 3 * 10 (kills) = 80.
    assert_eq!(env.money(), 80);
}

#[test]
fn economy_blocks_unaffordable_placements() {
    let env = GardogsEnv::new(GameConfig::test_minimal());
    // Two dogs cost 100; with 100 money exactly two placements are affordable.
    let p1 = env.place_index(0, (0, 0)).unwrap();
    let p2 = env.place_index(0, (1, 0)).unwrap();
    let p3 = env.place_index(0, (2, 0)).unwrap();

    let mut env = env;
    assert!(env.action_mask()[p1]);
    env.step(env.decode(p1));
    assert_eq!(env.money(), 50);

    assert!(env.action_mask()[p2], "second dog still affordable at 50");
    env.step(env.decode(p2));
    assert_eq!(env.money(), 0);

    assert!(
        !env.action_mask()[p3],
        "no money left: further placements illegal"
    );
}

#[test]
fn cannot_build_on_occupied_cell() {
    let mut env = GardogsEnv::new(GameConfig::test_minimal());
    let cell = env.place_index(0, (0, 0)).unwrap();
    env.step(env.decode(cell));
    assert!(
        !env.action_mask()[cell],
        "an occupied cell must not be buildable again"
    );
}

#[test]
fn full_random_game_runs_end_to_end() {
    use gardogs_agent_stub::RandomLike;
    // A self-contained random-ish driver (no agent crate dependency in env tests):
    let mut env = GardogsEnv::new(GameConfig::phase1());
    env.reset(123);
    let mut rng = RandomLike::new(123);
    let mut steps = 0;
    loop {
        let mask = env.action_mask();
        let legal: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| m.then_some(i))
            .collect();
        let idx = legal[rng.next() as usize % legal.len()];
        let sr = env.step(env.decode(idx));
        steps += 1;
        if sr.done {
            break;
        }
        assert!(steps < 100_000, "random game failed to terminate");
    }
    assert!(env.is_done());
}

/// Tiny deterministic PRNG so the env test suite stays dependency-free.
mod gardogs_agent_stub {
    pub struct RandomLike {
        state: u64,
    }
    impl RandomLike {
        pub fn new(seed: u64) -> Self {
            RandomLike {
                state: seed.wrapping_add(0x9E3779B97F4A7C15),
            }
        }
        pub fn next(&mut self) -> u64 {
            // xorshift64*
            let mut x = self.state;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.state = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
    }
}
