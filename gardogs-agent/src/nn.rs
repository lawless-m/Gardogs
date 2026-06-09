//! A minimal multilayer perceptron with manual backprop and an Adam optimiser.
//!
//! Deliberately dependency-free and deterministic: weights initialise from a
//! seeded RNG and every operation is plain `f32` arithmetic. It is sized for the
//! Phase 1 compact-vector observation (tens of inputs, ~128-unit hidden layers,
//! one output per action). ReLU between hidden layers; the output layer is
//! linear (it produces Q-values). See `04-agent-and-training.md`.

use std::io::{self, Read, Write};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Adam hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for Adam {
    fn default() -> Self {
        Adam {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// One fully-connected layer: `y = W x + b`, plus gradient and Adam state.
struct Linear {
    in_dim: usize,
    out_dim: usize,
    w: Vec<f32>, // out_dim * in_dim, row-major
    b: Vec<f32>, // out_dim
    gw: Vec<f32>,
    gb: Vec<f32>,
    mw: Vec<f32>,
    vw: Vec<f32>,
    mb: Vec<f32>,
    vb: Vec<f32>,
}

impl Linear {
    fn new(in_dim: usize, out_dim: usize, rng: &mut ChaCha8Rng) -> Self {
        // Kaiming/He init for ReLU: N(0, sqrt(2/in_dim)).
        let std = (2.0 / in_dim as f32).sqrt();
        let w = (0..out_dim * in_dim).map(|_| normal(rng) * std).collect();
        Linear {
            in_dim,
            out_dim,
            w,
            b: vec![0.0; out_dim],
            gw: vec![0.0; out_dim * in_dim],
            gb: vec![0.0; out_dim],
            mw: vec![0.0; out_dim * in_dim],
            vw: vec![0.0; out_dim * in_dim],
            mb: vec![0.0; out_dim],
            vb: vec![0.0; out_dim],
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = self.b.clone();
        for o in 0..self.out_dim {
            let row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
            let mut acc = y[o];
            for i in 0..self.in_dim {
                acc += row[i] * x[i];
            }
            y[o] = acc;
        }
        y
    }

    fn zero_grad(&mut self) {
        self.gw.iter_mut().for_each(|g| *g = 0.0);
        self.gb.iter_mut().for_each(|g| *g = 0.0);
    }

    /// Adam update using the accumulated gradients (already mean-reduced).
    /// `t` is the optimiser step count (1-based) for bias correction.
    fn adam_step(&mut self, hp: Adam, t: u64) {
        let bc1 = 1.0 - hp.beta1.powi(t as i32);
        let bc2 = 1.0 - hp.beta2.powi(t as i32);
        for k in 0..self.w.len() {
            self.mw[k] = hp.beta1 * self.mw[k] + (1.0 - hp.beta1) * self.gw[k];
            self.vw[k] = hp.beta2 * self.vw[k] + (1.0 - hp.beta2) * self.gw[k] * self.gw[k];
            let mhat = self.mw[k] / bc1;
            let vhat = self.vw[k] / bc2;
            self.w[k] -= hp.lr * mhat / (vhat.sqrt() + hp.eps);
        }
        for o in 0..self.b.len() {
            self.mb[o] = hp.beta1 * self.mb[o] + (1.0 - hp.beta1) * self.gb[o];
            self.vb[o] = hp.beta2 * self.vb[o] + (1.0 - hp.beta2) * self.gb[o] * self.gb[o];
            let mhat = self.mb[o] / bc1;
            let vhat = self.vb[o] / bc2;
            self.b[o] -= hp.lr * mhat / (vhat.sqrt() + hp.eps);
        }
    }
}

/// Cached activations from a training forward pass, needed for backprop.
pub struct Cache {
    /// Input fed to each layer (`input[0]` is the network input).
    input: Vec<Vec<f32>>,
    /// Pre-activation of each layer.
    z: Vec<Vec<f32>>,
    /// Network output (the last layer's pre-activation; output is linear).
    pub out: Vec<f32>,
}

/// A stack of linear layers with ReLU between them (linear output).
pub struct Mlp {
    layers: Vec<Linear>,
    t: u64,
}

impl Mlp {
    /// Build an MLP with the given layer sizes, e.g. `[obs, 128, 128, actions]`.
    pub fn new(sizes: &[usize], rng: &mut ChaCha8Rng) -> Self {
        assert!(sizes.len() >= 2, "need at least an input and output size");
        let layers = sizes
            .windows(2)
            .map(|w| Linear::new(w[0], w[1], rng))
            .collect();
        Mlp { layers, t: 0 }
    }

    pub fn input_dim(&self) -> usize {
        self.layers[0].in_dim
    }
    pub fn output_dim(&self) -> usize {
        self.layers.last().unwrap().out_dim
    }

    /// Inference: Q-values for one observation. ReLU on hidden layers only.
    pub fn q(&self, x: &[f32]) -> Vec<f32> {
        let mut a = x.to_vec();
        for (l, layer) in self.layers.iter().enumerate() {
            let z = layer.forward(&a);
            a = if l + 1 < self.layers.len() {
                relu(&z)
            } else {
                z
            };
        }
        a
    }

    /// Forward pass that caches activations for [`Mlp::backward`].
    pub fn forward_train(&self, x: &[f32]) -> Cache {
        let mut input = Vec::with_capacity(self.layers.len());
        let mut z = Vec::with_capacity(self.layers.len());
        let mut a = x.to_vec();
        for (l, layer) in self.layers.iter().enumerate() {
            input.push(a.clone());
            let zl = layer.forward(&a);
            a = if l + 1 < self.layers.len() {
                relu(&zl)
            } else {
                zl.clone()
            };
            z.push(zl);
        }
        let out = a;
        Cache { input, z, out }
    }

    /// Accumulate gradients for one sample, given `dL/d(out)`. Call
    /// [`Mlp::zero_grad`] before a batch and [`Mlp::adam_step`] after it.
    pub fn backward(&mut self, cache: &Cache, grad_out: &[f32]) {
        let mut grad = grad_out.to_vec(); // dL/dz of the current (initially last) layer
        for l in (0..self.layers.len()).rev() {
            let (in_dim, out_dim) = (self.layers[l].in_dim, self.layers[l].out_dim);
            let inp = &cache.input[l];

            {
                let layer = &mut self.layers[l];
                for o in 0..out_dim {
                    layer.gb[o] += grad[o];
                    let base = o * in_dim;
                    let go = grad[o];
                    for i in 0..in_dim {
                        layer.gw[base + i] += go * inp[i];
                    }
                }
            }

            if l > 0 {
                // Propagate to the previous layer's activation, then through ReLU.
                let layer = &self.layers[l];
                let mut da = vec![0.0f32; in_dim];
                for o in 0..out_dim {
                    let base = o * in_dim;
                    let go = grad[o];
                    for i in 0..in_dim {
                        da[i] += layer.w[base + i] * go;
                    }
                }
                let zprev = &cache.z[l - 1];
                for i in 0..in_dim {
                    da[i] *= if zprev[i] > 0.0 { 1.0 } else { 0.0 };
                }
                grad = da;
            }
        }
    }

    pub fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }

    /// Mean-reduce the accumulated gradients over `batch` samples, optionally
    /// clip them to a global L2 norm (a key DQN stabiliser), and step Adam.
    pub fn adam_step(&mut self, hp: Adam, batch: usize, clip_norm: Option<f32>) {
        let scale = 1.0 / batch as f32;
        for layer in &mut self.layers {
            layer.gw.iter_mut().for_each(|g| *g *= scale);
            layer.gb.iter_mut().for_each(|g| *g *= scale);
        }

        if let Some(max_norm) = clip_norm {
            let mut sumsq = 0.0f32;
            for layer in &self.layers {
                sumsq += layer.gw.iter().map(|g| g * g).sum::<f32>();
                sumsq += layer.gb.iter().map(|g| g * g).sum::<f32>();
            }
            let norm = sumsq.sqrt();
            if norm > max_norm {
                let factor = max_norm / (norm + 1e-6);
                for layer in &mut self.layers {
                    layer.gw.iter_mut().for_each(|g| *g *= factor);
                    layer.gb.iter_mut().for_each(|g| *g *= factor);
                }
            }
        }

        self.t += 1;
        let t = self.t;
        for layer in &mut self.layers {
            layer.adam_step(hp, t);
        }
    }

    /// Copy weights and biases from `other` (used to sync the target network).
    pub fn copy_weights_from(&mut self, other: &Mlp) {
        for (dst, src) in self.layers.iter_mut().zip(&other.layers) {
            dst.w.copy_from_slice(&src.w);
            dst.b.copy_from_slice(&src.b);
        }
    }

    // --- checkpointing (a small, explicit binary format; no extra deps) --------

    pub fn save<W: Write>(&self, mut out: W) -> io::Result<()> {
        out.write_all(b"GDQN")?;
        write_u32(&mut out, self.layers.len() as u32)?;
        for layer in &self.layers {
            write_u32(&mut out, layer.in_dim as u32)?;
            write_u32(&mut out, layer.out_dim as u32)?;
            write_f32s(&mut out, &layer.w)?;
            write_f32s(&mut out, &layer.b)?;
        }
        Ok(())
    }

    pub fn load<R: Read>(mut inp: R) -> io::Result<Mlp> {
        let mut magic = [0u8; 4];
        inp.read_exact(&mut magic)?;
        if &magic != b"GDQN" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad checkpoint magic",
            ));
        }
        let n = read_u32(&mut inp)? as usize;
        // Weights are overwritten from the file; the RNG only sizes the buffers.
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            let in_dim = read_u32(&mut inp)? as usize;
            let out_dim = read_u32(&mut inp)? as usize;
            let mut layer = Linear::new(in_dim, out_dim, &mut rng);
            read_f32s(&mut inp, &mut layer.w)?;
            read_f32s(&mut inp, &mut layer.b)?;
            layers.push(layer);
        }
        Ok(Mlp { layers, t: 0 })
    }
}

fn relu(z: &[f32]) -> Vec<f32> {
    z.iter().map(|&v| v.max(0.0)).collect()
}

/// One sample from a standard normal via Box–Muller (keeps us dependency-free).
fn normal(rng: &mut ChaCha8Rng) -> f32 {
    let u1 = 1.0 - rng.gen::<f32>(); // in (0, 1]
    let u2 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

// Small helpers for the checkpoint format.
fn write_u32<W: Write>(out: &mut W, v: u32) -> io::Result<()> {
    out.write_all(&v.to_le_bytes())
}
fn read_u32<R: Read>(inp: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    inp.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn write_f32s<W: Write>(out: &mut W, xs: &[f32]) -> io::Result<()> {
    for &x in xs {
        out.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}
fn read_f32s<R: Read>(inp: &mut R, xs: &mut [f32]) -> io::Result<()> {
    let mut b = [0u8; 4];
    for x in xs.iter_mut() {
        inp.read_exact(&mut b)?;
        *x = f32::from_le_bytes(b);
    }
    Ok(())
}
