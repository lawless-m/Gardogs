//! Thin driver binary for Gardogs.
//!
//! Phase 0 provides one verb, `play`: run a full Phase-1 game headless with the
//! random agent and print a summary, proving the `reset`/`step` loop end to end.
//! `train` / `eval` / `watch` (`01-architecture.md`) arrive in later phases.

use std::process::ExitCode;

use gardogs_agent::{Policy, RandomAgent};
use gardogs_env::{Env, GameConfig, GardogsEnv};

struct Args {
    seed: u64,
    agent_seed: u64,
    max_ticks: u64,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        seed: 0,
        agent_seed: 0,
        max_ticks: 100_000,
    };
    let mut it = std::env::args().skip(1).peekable();

    // Optional leading verb; only `play` is supported in Phase 0.
    if let Some(first) = it.peek() {
        if !first.starts_with('-') {
            let verb = it.next().unwrap();
            if verb != "play" {
                return Err(format!(
                    "unknown command '{verb}' (only 'play' is supported)"
                ));
            }
        }
    }

    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .ok_or_else(|| format!("missing value for {flag}"))
                .and_then(|v| {
                    v.parse::<u64>()
                        .map_err(|_| format!("invalid number for {flag}: {v}"))
                })
        };
        match flag.as_str() {
            "--seed" => args.seed = value()?,
            "--agent-seed" => args.agent_seed = value()?,
            "--max-ticks" => args.max_ticks = value()?,
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(args)
}

fn usage() {
    eprintln!(
        "gardogs play [--seed N] [--agent-seed N] [--max-ticks N]\n\
         \n\
         Runs one full Phase-1 game with the random agent and prints a summary."
    );
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if e != "help" {
                eprintln!("error: {e}\n");
            }
            usage();
            return if e == "help" {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            };
        }
    };

    let mut env = GardogsEnv::new(GameConfig::phase1());
    let mut agent = RandomAgent::new(args.agent_seed);
    let mut obs = env.reset(args.seed);

    let mut steps = 0u64;
    let (info, done) = loop {
        let mask = env.action_mask();
        let action_idx = agent.act(&obs, &mask);
        let action = env.decode(action_idx);
        let sr = env.step(action);
        obs = sr.obs;
        steps += 1;
        if sr.done || steps >= args.max_ticks {
            break (sr.info, sr.done);
        }
    };

    let outcome = if info.won {
        "WIN"
    } else if done {
        "LOSE"
    } else {
        "TIMEOUT"
    };

    println!("seed={} agent_seed={}", args.seed, args.agent_seed);
    println!(
        "ticks={} wave={} score={:.1} money={} lives={} -> {}",
        info.tick, info.wave, info.score, info.money, info.lives, outcome
    );

    ExitCode::SUCCESS
}
