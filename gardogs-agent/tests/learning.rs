//! Phase 1 tests: backprop correctness, training determinism, and that the DQN
//! actually learns a trivial defence (the Phase 1 "done" criterion in miniature).

use gardogs_agent::{evaluate, train, Adam, DqnConfig, Mlp, TrainConfig};
use gardogs_env::GameConfig;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// The MLP can fit XOR — exercises forward, backward and the Adam step end to end.
/// If backprop is wrong, this won't converge.
#[test]
fn mlp_learns_xor() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut net = Mlp::new(&[2, 16, 16, 1], &mut rng);
    let hp = Adam {
        lr: 0.01,
        ..Adam::default()
    };

    let data = [
        ([0.0f32, 0.0], 0.0f32),
        ([1.0, 0.0], 1.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    let mut last_loss = f32::INFINITY;
    for _ in 0..3000 {
        net.zero_grad();
        let mut loss = 0.0f32;
        for (x, target) in &data {
            let cache = net.forward_train(x);
            let e = cache.out[0] - target; // dL/dout for 0.5*e^2
            loss += 0.5 * e * e;
            net.backward(&cache, &[e]);
        }
        net.adam_step(hp, data.len(), None);
        last_loss = loss / data.len() as f32;
    }

    assert!(
        last_loss < 0.02,
        "XOR loss should be small, got {last_loss}"
    );
    for (x, target) in &data {
        let pred = net.q(x)[0];
        assert!(
            (pred - target).abs() < 0.25,
            "XOR({x:?}) ≈ {pred:.3}, expected {target}"
        );
    }
}

/// A checkpoint round-trips: a loaded network reproduces the saved one's outputs.
#[test]
fn checkpoint_roundtrip() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let net = Mlp::new(&[5, 8, 3], &mut rng);
    let probe = [0.1f32, -0.2, 0.3, 0.4, -0.5];
    let before = net.q(&probe);

    let mut buf = Vec::new();
    net.save(&mut buf).unwrap();
    let loaded = Mlp::load(&buf[..]).unwrap();
    let after = loaded.q(&probe);

    assert_eq!(
        before, after,
        "loaded network must reproduce outputs exactly"
    );
}

/// Two runs with the same seed must produce identical results (reproducibility).
#[test]
fn training_is_deterministic() {
    let cfg = small_dqn_cfg();
    let tcfg = TrainConfig {
        episodes: 30,
        eval_every: 0,
        eval_games: 0,
        verbose: false,
        ..TrainConfig::default()
    };

    let (a1, r1) = train(GameConfig::test_minimal(), cfg.clone(), tcfg.clone(), 42);
    let (a2, r2) = train(GameConfig::test_minimal(), cfg, tcfg, 42);

    assert_eq!(
        r1.episode_rewards, r2.episode_rewards,
        "reward curves must match"
    );
    // And the learned policies must be identical.
    let obs = vec![0.0f32; obs_dim()];
    assert_eq!(
        a1.q_values(&obs),
        a2.q_values(&obs),
        "Q-functions must match"
    );
}

/// The agent learns the trivial defence: place the dog and win. This is the
/// Phase 1 milestone shrunk to a fast, deterministic scenario.
#[test]
fn dqn_learns_to_defend_minimal() {
    let env_cfg = GameConfig::test_minimal();
    let tcfg = TrainConfig {
        episodes: 150,
        eval_every: 0,
        eval_games: 0,
        verbose: false,
        ..TrainConfig::default()
    };

    let (agent, _report) = train(env_cfg.clone(), small_dqn_cfg(), tcfg, 0);
    let ev = evaluate(&agent, &env_cfg, 20, 5_000_000);

    // Random play wins this essentially never; a learned policy should win often.
    assert!(
        ev.win_rate >= 0.8,
        "expected the agent to learn to win, got win_rate {:.2}",
        ev.win_rate
    );
}

/// A small, fast DQN config so the tests run quickly even in debug builds.
fn small_dqn_cfg() -> DqnConfig {
    DqnConfig {
        hidden: vec![32],
        adam: Adam {
            lr: 1e-3,
            ..Adam::default()
        },
        gamma: 0.99,
        batch: 16,
        buffer_cap: 5_000,
        target_update_every: 150,
        eps_start: 1.0,
        eps_end: 0.05,
        eps_decay_steps: 1_500,
        warmup: 100,
        train_every: 1,
        grad_clip: Some(10.0),
    }
}

fn obs_dim() -> usize {
    use gardogs_env::{Env, GardogsEnv, ObsShape};
    let env = GardogsEnv::new(GameConfig::test_minimal());
    match env.observation_shape() {
        ObsShape::Flat(n) => n,
        ObsShape::Planes {
            channels,
            height,
            width,
        } => channels * height * width,
    }
}
