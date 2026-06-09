//! Thin driver binary for Gardogs.
//!
//! Verbs:
//!   - `play`  — run a full game with the random agent (Phase 0 smoke test).
//!   - `train` — train a DQN on the chosen rules; optionally save a checkpoint.
//!   - `eval`  — play games greedily with a saved checkpoint and report metrics.
//!
//! `watch` (the live viewer) arrives in Phase 4.

use std::process::ExitCode;

use gardogs_agent::{dog_usage, evaluate, train, Dqn, DqnConfig, Policy, RandomAgent, TrainConfig};
use gardogs_env::{Env, GameConfig, GardogsEnv};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (verb, rest) = match args.split_first() {
        Some((v, r)) if !v.starts_with('-') => (v.as_str(), r),
        _ => ("play", &args[..]), // default verb
    };

    let result = match verb {
        "play" => cmd_play(rest),
        "train" => cmd_train(rest),
        "eval" => cmd_eval(rest),
        "-h" | "--help" | "help" => {
            usage();
            return ExitCode::SUCCESS;
        }
        other => Err(format!("unknown command '{other}'")),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}\n");
            usage();
            ExitCode::from(2)
        }
    }
}

fn usage() {
    eprintln!(
        "gardogs <command> [options]\n\
         \n\
         commands:\n  \
           play   [--config phase1|phase2] [--seed N] [--agent-seed N] [--max-ticks N]\n  \
           train  [--config phase1|phase2] [--seed N] [--episodes N] [--out PATH] [--log PATH] [--quiet]\n  \
           eval   --checkpoint PATH [--config phase1|phase2] [--games N] [--seed N]\n\
         \n\
         --config defaults to phase2.\n"
    );
}

/// Select a built-in game config by name.
fn config_by_name(name: &str) -> Result<GameConfig, String> {
    match name {
        "phase1" => Ok(GameConfig::phase1()),
        "phase2" => Ok(GameConfig::phase2()),
        other => Err(format!(
            "unknown --config '{other}' (expected phase1 or phase2)"
        )),
    }
}

/// Minimal flag parser: pulls `--flag value` pairs into a small lookup.
struct Flags {
    map: std::collections::HashMap<String, String>,
}

impl Flags {
    fn parse(args: &[String]) -> Result<Flags, String> {
        let mut map = std::collections::HashMap::new();
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            let key = flag
                .strip_prefix("--")
                .ok_or_else(|| format!("expected a flag, got '{flag}'"))?;
            if key == "quiet" {
                map.insert(key.to_string(), "1".to_string());
            } else {
                let val = it
                    .next()
                    .ok_or_else(|| format!("missing value for --{key}"))?;
                map.insert(key.to_string(), val.clone());
            }
        }
        Ok(Flags { map })
    }

    fn get_u64(&self, key: &str, default: u64) -> Result<u64, String> {
        match self.map.get(key) {
            Some(v) => v
                .parse()
                .map_err(|_| format!("invalid number for --{key}: {v}")),
            None => Ok(default),
        }
    }
    fn get_str(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(String::as_str)
    }
    fn has(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }
    fn config(&self) -> Result<GameConfig, String> {
        config_by_name(self.get_str("config").unwrap_or("phase2"))
    }
}

fn cmd_play(args: &[String]) -> Result<(), String> {
    let f = Flags::parse(args)?;
    let seed = f.get_u64("seed", 0)?;
    let agent_seed = f.get_u64("agent-seed", 0)?;
    let max_ticks = f.get_u64("max-ticks", 100_000)?;

    let mut env = GardogsEnv::new(f.config()?);
    let mut agent = RandomAgent::new(agent_seed);
    let mut obs = env.reset(seed);

    let mut steps = 0u64;
    let (info, done) = loop {
        let mask = env.action_mask();
        let action = env.decode(agent.act(&obs, &mask));
        let sr = env.step(action);
        obs = sr.obs;
        steps += 1;
        if sr.done || steps >= max_ticks {
            break (sr.info, sr.done);
        }
    };

    let outcome = outcome_str(info.won, done);
    println!("seed={seed} agent_seed={agent_seed}");
    println!(
        "ticks={} wave={} score={:.1} money={} lives={} -> {}",
        info.tick, info.wave, info.score, info.money, info.lives, outcome
    );
    Ok(())
}

fn cmd_train(args: &[String]) -> Result<(), String> {
    let f = Flags::parse(args)?;
    let seed = f.get_u64("seed", 0)?;
    let episodes = f.get_u64("episodes", 2_000)? as usize;
    let out = f.get_str("out").map(str::to_string);
    let log = f.get_str("log").map(str::to_string);

    let cfg_name = f.get_str("config").unwrap_or("phase2").to_string();
    let env_cfg = f.config()?;

    // Phase 2 is a harder exploration problem (161 actions, sparse delayed
    // reward), so it gets longer exploration and a small damage-shaping signal;
    // Phase 1 keeps the plain defaults that already win.
    let (dqn_cfg, shaping) = match cfg_name.as_str() {
        "phase2" => (
            DqnConfig {
                eps_decay_steps: 150_000,
                ..DqnConfig::default()
            },
            0.05,
        ),
        _ => (DqnConfig::default(), 0.0),
    };

    let train_cfg = TrainConfig {
        episodes,
        verbose: !f.has("quiet"),
        log_path: log,
        reward_shaping: shaping,
        ..TrainConfig::default()
    };

    eprintln!("training DQN on {cfg_name} (seed={seed}, episodes={episodes})...");
    let (agent, report) = train(env_cfg.clone(), dqn_cfg, train_cfg, seed);

    let final_eval = evaluate(&agent, &env_cfg, 100, 2_000_000);
    println!(
        "final (greedy, 100 games): win {:.1}% | avg_reward {:.2} | avg_waves {:.2}",
        final_eval.win_rate * 100.0,
        final_eval.avg_reward,
        final_eval.avg_waves
    );

    let first = report.episode_rewards.first().copied().unwrap_or(0.0);
    let last = report
        .episode_rewards
        .iter()
        .rev()
        .take(50)
        .copied()
        .sum::<f32>()
        / 50.0_f32.min(report.episode_rewards.len() as f32);
    println!("episode reward: first={first:.1} -> last50_avg={last:.1}");

    // What the trained agent actually fields (evidence of differentiated play).
    let usage = dog_usage(&agent, &env_cfg, 2_000_000);
    let composition: Vec<String> = env_cfg
        .dogs
        .iter()
        .zip(&usage)
        .filter(|(_, &n)| n > 0)
        .map(|(d, n)| format!("{}×{n}", d.name))
        .collect();
    if !composition.is_empty() {
        println!("dog usage (one greedy game): {}", composition.join(", "));
    }

    if let Some(path) = out {
        agent
            .save(&path)
            .map_err(|e| format!("failed to save checkpoint: {e}"))?;
        println!("saved checkpoint to {path}");
    }
    Ok(())
}

fn cmd_eval(args: &[String]) -> Result<(), String> {
    let f = Flags::parse(args)?;
    let checkpoint = f
        .get_str("checkpoint")
        .ok_or("eval requires --checkpoint PATH")?;
    let games = f.get_u64("games", 100)? as usize;
    let seed = f.get_u64("seed", 2_000_000)?;

    let agent = Dqn::load_for_eval(checkpoint)
        .map_err(|e| format!("failed to load checkpoint '{checkpoint}': {e}"))?;
    let ev = evaluate(&agent, &f.config()?, games, seed);
    println!(
        "eval ({games} games): win {:.1}% | avg_reward {:.2} | avg_waves {:.2}",
        ev.win_rate * 100.0,
        ev.avg_reward,
        ev.avg_waves
    );
    Ok(())
}

fn outcome_str(won: bool, done: bool) -> &'static str {
    if won {
        "WIN"
    } else if done {
        "LOSE"
    } else {
        "TIMEOUT"
    }
}
