//! The training loop and greedy evaluation that tie the DQN to the environment.
//!
//! Everything is seeded so a run reproduces exactly: the agent's init/sampling
//! seed, and a per-episode environment seed derived from the run seed. See the
//! "training loop" and "reproducibility" sections of `04-agent-and-training.md`.

use std::io::Write;

use gardogs_env::{Action, Env, GameConfig, GardogsEnv};

use crate::dqn::{Dqn, DqnConfig};
use crate::replay::Transition;

/// How a training run is driven (orthogonal to the DQN hyperparameters).
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub episodes: usize,
    /// Run a greedy evaluation every this many episodes (0 disables it).
    pub eval_every: usize,
    pub eval_games: usize,
    /// Safety cap so a single episode can never hang the run.
    pub max_steps_per_episode: u64,
    /// Print progress to stderr.
    pub verbose: bool,
    /// Optional CSV reward log (`episode,env_step,epsilon,episode_reward`).
    pub log_path: Option<String>,
    /// Optional reward shaping: add `reward_shaping * damage_dealt` to the reward
    /// the agent *learns* from (densifies the signal for hard exploration). The
    /// reported episode reward and all evaluation stay on the canonical reward,
    /// so metrics remain comparable. 0.0 disables it. See `04-agent-and-training.md`.
    pub reward_shaping: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            episodes: 2_000,
            eval_every: 50,
            eval_games: 30,
            max_steps_per_episode: 5_000,
            verbose: true,
            log_path: None,
            reward_shaping: 0.0,
        }
    }
}

/// The result of one greedy evaluation.
#[derive(Clone, Copy, Debug)]
pub struct Eval {
    pub avg_reward: f32,
    pub win_rate: f32,
    pub avg_waves: f32,
}

/// What a training run produced, for plotting / regression checks.
#[derive(Clone, Debug, Default)]
pub struct TrainReport {
    pub episode_rewards: Vec<f32>,
    pub evals: Vec<(usize, Eval)>, // (episode, eval)
}

/// Train a fresh agent on `env_cfg`. Returns the trained agent and a report.
pub fn train(
    env_cfg: GameConfig,
    dqn_cfg: DqnConfig,
    train_cfg: TrainConfig,
    seed: u64,
) -> (Dqn, TrainReport) {
    let mut env = GardogsEnv::new(env_cfg.clone());
    let obs_dim = match env.observation_shape() {
        gardogs_env::ObsShape::Flat(n) => n,
        gardogs_env::ObsShape::Planes {
            channels,
            height,
            width,
        } => channels * height * width,
    };
    let mut agent = Dqn::new(obs_dim, env.action_count(), dqn_cfg, seed);

    let mut log = train_cfg.log_path.as_ref().map(|p| {
        let mut f = std::fs::File::create(p).expect("cannot create reward log");
        writeln!(f, "episode,env_step,epsilon,episode_reward").ok();
        f
    });

    let mut report = TrainReport::default();
    // Training is noisy and DQN can transiently forget; keep the best-evaluated
    // network and restore it at the end so the returned agent is the best seen.
    let mut best: Option<(Eval, Vec<u8>)> = None;

    for episode in 0..train_cfg.episodes {
        // Distinct, reproducible environment per episode.
        let mut obs = env.reset(seed.wrapping_add(episode as u64));
        let mut mask = env.action_mask();
        let mut ep_reward = 0.0f32;
        let mut steps = 0u64;

        loop {
            let action_idx = agent.act_explore(&obs.data, &mask);
            let sr = env.step(env.decode(action_idx));
            let next_mask = env.action_mask();

            // The agent learns from an optionally-shaped reward; metrics use the
            // canonical one (`ep_reward`), so evaluation stays comparable.
            let learn_reward = sr.reward + train_cfg.reward_shaping * sr.info.damage_dealt;
            agent.observe(Transition {
                state: std::mem::take(&mut obs.data),
                action: action_idx,
                reward: learn_reward,
                next_state: sr.obs.data.clone(),
                done: sr.done,
                next_mask: next_mask.clone(),
            });

            ep_reward += sr.reward;
            obs = sr.obs;
            mask = next_mask;
            steps += 1;
            if sr.done || steps >= train_cfg.max_steps_per_episode {
                break;
            }
        }

        report.episode_rewards.push(ep_reward);
        if let Some(f) = log.as_mut() {
            writeln!(
                f,
                "{},{},{:.4},{:.3}",
                episode,
                agent.env_steps(),
                agent.epsilon(),
                ep_reward
            )
            .ok();
        }

        let do_eval = train_cfg.eval_every > 0 && (episode + 1) % train_cfg.eval_every == 0
            || episode + 1 == train_cfg.episodes;
        if do_eval {
            let ev = evaluate(&agent, &env_cfg, train_cfg.eval_games, 1_000_000);
            report.evals.push((episode, ev));

            let better = match &best {
                None => true,
                Some((b, _)) => eval_key(&ev) > eval_key(b),
            };
            if better {
                best = Some((ev, agent.snapshot()));
            }

            if train_cfg.verbose {
                let recent = recent_mean(&report.episode_rewards, train_cfg.eval_every.max(1));
                eprintln!(
                    "ep {:>5} | steps {:>7} | eps {:.2} | train_r(avg) {:>7.2} | \
                     eval win {:>5.1}% avg_r {:>7.2} waves {:.2}",
                    episode + 1,
                    agent.env_steps(),
                    agent.epsilon(),
                    recent,
                    ev.win_rate * 100.0,
                    ev.avg_reward,
                    ev.avg_waves,
                );
            }
        }
    }

    // Restore the best-evaluated weights, if any evaluation happened.
    if let Some((ev, bytes)) = &best {
        agent.restore(bytes);
        if train_cfg.verbose {
            eprintln!(
                "restored best-evaluated network: win {:.1}% avg_r {:.2}",
                ev.win_rate * 100.0,
                ev.avg_reward
            );
        }
    }

    (agent, report)
}

/// Rank evaluations by win rate first, then average reward.
fn eval_key(e: &Eval) -> (i32, i32) {
    ((e.win_rate * 1000.0) as i32, (e.avg_reward * 100.0) as i32)
}

/// Play `games` greedily (no exploration) and report averages. Uses a fixed seed
/// base distinct from training so evaluations are comparable across checkpoints.
pub fn evaluate(agent: &Dqn, env_cfg: &GameConfig, games: usize, seed_base: u64) -> Eval {
    let mut env = GardogsEnv::new(env_cfg.clone());
    let (mut total_reward, mut wins, mut total_waves) = (0.0f32, 0usize, 0u32);

    for g in 0..games {
        let mut obs = env.reset(seed_base.wrapping_add(g as u64));
        let mut ep_reward = 0.0f32;
        let mut steps = 0u64;
        loop {
            let mask = env.action_mask();
            let a = agent.act_greedy(&obs.data, &mask);
            let sr = env.step(env.decode(a));
            ep_reward += sr.reward;
            obs = sr.obs;
            steps += 1;
            if sr.done {
                if sr.info.won {
                    wins += 1;
                }
                total_waves += sr.info.wave as u32;
                break;
            }
            if steps >= 5_000 {
                total_waves += sr.info.wave as u32;
                break;
            }
        }
        total_reward += ep_reward;
    }

    let n = games.max(1) as f32;
    Eval {
        avg_reward: total_reward / n,
        win_rate: wins as f32 / n,
        avg_waves: total_waves as f32 / n,
    }
}

/// Play one greedy game and count how many of each dog type the agent placed.
/// Evidence of *differentiated* play (Phase 2): a learned agent should field
/// air-capable dogs against seagulls, not spam one breed.
pub fn dog_usage(agent: &Dqn, env_cfg: &GameConfig, seed: u64) -> Vec<u32> {
    let mut env = GardogsEnv::new(env_cfg.clone());
    let mut obs = env.reset(seed);
    let mut counts = vec![0u32; env_cfg.dogs.len()];
    let mut steps = 0u64;
    loop {
        let mask = env.action_mask();
        let a = agent.act_greedy(&obs.data, &mask);
        // Greedy actions come from the mask, so a chosen placement always succeeds.
        if let Action::Place { dog, .. } = env.decode(a) {
            counts[dog] += 1;
        }
        let sr = env.step(env.decode(a));
        obs = sr.obs;
        steps += 1;
        if sr.done || steps >= 5_000 {
            break;
        }
    }
    counts
}

fn recent_mean(xs: &[f32], window: usize) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let start = xs.len().saturating_sub(window);
    let slice = &xs[start..];
    slice.iter().sum::<f32>() / slice.len() as f32
}
