//! Phase 2 mechanics: differentiated threats. These pin down the rules the agent
//! must learn to exploit — air enemies need air-capable dogs, and the collie
//! slows what passes it — independently of any learning.

use gardogs_env::{
    DogSpec, EnemySpec, Env, GameConfig, GardogsEnv, GridConfig, Movement, SpawnGroup, Target,
    WaveSpec,
};

/// A one-dog, one-enemy scenario for isolating a single rule.
fn duel(dog: DogSpec, enemy: EnemySpec, count: u32, start_lives: u32) -> GameConfig {
    GameConfig {
        grid: GridConfig {
            width: 8,
            height: 6,
            path_row: 3,
        },
        tick_rate: 10.0,
        start_money: 200,
        start_lives,
        breather_s: 1.0,
        dogs: vec![dog],
        enemies: vec![enemy],
        waves: vec![WaveSpec {
            groups: vec![SpawnGroup {
                enemy: 0,
                count,
                spacing_s: 2.0,
                start_s: 0.0,
            }],
        }],
        max_tracked_enemies: 8,
    }
}

fn seagull() -> EnemySpec {
    EnemySpec {
        name: "Seagull",
        health: 15.0,
        speed: 1.5,
        movement: Movement::Air,
        kill_reward: 8.0,
        kill_money: 12,
        lives_cost: 1,
    }
}

fn cat() -> EnemySpec {
    EnemySpec {
        name: "Cat",
        health: 20.0,
        speed: 1.0,
        movement: Movement::Ground,
        kill_reward: 5.0,
        kill_money: 10,
        lives_cost: 1,
    }
}

/// A fast one-shot dog with the given targeting (so kills are unambiguous).
fn gun(target: Target) -> DogSpec {
    DogSpec {
        name: "Gun",
        cost: 50,
        damage: 50.0,
        range: 2.0,
        fire_rate: 2.0,
        target: Some(target),
        slow_factor: None,
    }
}

fn run_to_end(mut env: GardogsEnv) -> GardogsEnv {
    loop {
        let sr = env.step(gardogs_env::Action::Noop);
        if sr.done {
            break;
        }
        assert!(env.tick() < 100_000, "did not terminate");
    }
    env
}

#[test]
fn ground_dog_cannot_hit_air() {
    // A ground-only gun beside the path lets seagulls sail over the top.
    let mut env = GardogsEnv::new(duel(gun(Target::Ground), seagull(), 2, 3));
    let place = env.place_index(0, (1, 2)).unwrap();
    env.step(env.decode(place));
    let env = run_to_end(env);

    // Both seagulls leak (1 life each): the dog never touched them.
    assert_eq!(
        env.lives(),
        1,
        "ground dog should not stop airborne seagulls"
    );
}

#[test]
fn air_capable_dog_stops_seagulls() {
    // The same gun, but targeting `Both`, clears the air wave with no leaks.
    let mut env = GardogsEnv::new(duel(gun(Target::Both), seagull(), 2, 3));
    let place = env.place_index(0, (1, 2)).unwrap();
    env.step(env.decode(place));
    let env = run_to_end(env);

    assert!(env.won(), "an air-capable dog should win the air wave");
    assert_eq!(env.lives(), 3, "no seagull should have leaked");
}

#[test]
fn collie_slows_what_passes_it() {
    // A pure-support collie: same range, no damage, slows enemies in range.
    let collie = DogSpec {
        name: "Collie",
        cost: 80,
        damage: 0.0,
        range: 2.0,
        fire_rate: 0.0,
        target: None,
        slow_factor: Some(0.5),
    };

    // One cat, one life: the run ends when the cat leaks. With the collie beside
    // the path the cat is slowed and takes measurably longer to cross.
    let make = || GardogsEnv::new(duel(collie.clone(), cat(), 1, 1));

    let mut near = make();
    let p = near.place_index(0, (1, 2)).unwrap(); // adjacent to the path: in range
    near.step(near.decode(p));
    let near = run_to_end(near);

    let mut far = make();
    let p = far.place_index(0, (1, 0)).unwrap(); // 3 rows off the path: out of range
    far.step(far.decode(p));
    let far = run_to_end(far);

    assert!(
        near.tick() > far.tick(),
        "collie in range should slow the cat: near={} far={}",
        near.tick(),
        far.tick()
    );
}

#[test]
fn mastiffs_with_a_collie_kill_a_postman() {
    // Postmen (100 hp tanks) must be killable by the anti-tank combo: mastiffs
    // holding a chokepoint with a collie slowing the target so it stays in range.
    // This fails if the mastiff's range is too short to cover the path.
    let mastiff = DogSpec {
        name: "Mastiff",
        cost: 120,
        damage: 30.0,
        range: 1.5,
        fire_rate: 0.4,
        target: Some(Target::Ground),
        slow_factor: None,
    };
    let collie = DogSpec {
        name: "Collie",
        cost: 80,
        damage: 0.0,
        range: 2.0,
        fire_rate: 0.0,
        target: None,
        slow_factor: Some(0.5),
    };
    let postman = EnemySpec {
        name: "Postman",
        health: 100.0,
        speed: 0.5,
        movement: Movement::Ground,
        kill_reward: 15.0,
        kill_money: 25,
        lives_cost: 2,
    };
    let cfg = GameConfig {
        grid: GridConfig {
            width: 8,
            height: 6,
            path_row: 3,
        },
        tick_rate: 10.0,
        start_money: 400,
        start_lives: 1, // one postman through the door ends it
        breather_s: 1.0,
        dogs: vec![mastiff, collie],
        enemies: vec![postman],
        waves: vec![WaveSpec {
            groups: vec![SpawnGroup {
                enemy: 0,
                count: 1,
                spacing_s: 1.0,
                start_s: 0.0,
            }],
        }],
        max_tracked_enemies: 8,
    };

    let mut env = GardogsEnv::new(cfg);
    for (dog, cell) in [(0, (1, 2)), (0, (3, 2)), (1, (2, 2))] {
        let idx = env.place_index(dog, cell).unwrap();
        env.step(env.decode(idx));
    }
    let env = run_to_end(env);

    assert!(env.won(), "mastiffs + collie should bring down a postman");
    assert_eq!(env.lives(), 1, "the postman must not reach the door");
}

#[test]
fn phase2_observation_and_action_shapes() {
    use gardogs_env::ObsShape;
    let env = GardogsEnv::new(GameConfig::phase2());
    // path(48) + 4 dog planes(48*4) + 10 tracked * (2 + 3 enemy types) + 4 scalars
    //   = 48 + 192 + 50 + 4 = 294
    assert_eq!(env.observation_shape(), ObsShape::Flat(294));
    // NOOP + 4 dog types * 40 buildable cells
    assert_eq!(env.action_count(), 1 + 4 * 40);
}
