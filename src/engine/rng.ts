import { N_LAYERS, SIZES, type Params } from './params';

/** mulberry32: small seeded PRNG so every initialisation is reproducible from its seed. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Biases are drawn from [0, BIAS_MAX). */
export const BIAS_MAX = 0.5;

/**
 * Weights uniform(-1, 1) as in microtensor/nn.py.
 * Biases differ from microtensor (which starts them at 0): they start in [0, 0.5).
 * With zero biases the sample (0, 0) puts every first-layer z exactly on the ReLU kink,
 * where the derivative is not defined, and far more random draws start with dead neurons.
 */
export function initParams(seed: number): Params {
  const rand = mulberry32(seed);
  const W: number[][][] = [];
  const b: number[][] = [];
  for (let l = 0; l < N_LAYERS; l++) {
    const layer: number[][] = [];
    const biases: number[] = [];
    for (let j = 0; j < SIZES[l + 1]; j++) {
      const row: number[] = [];
      for (let k = 0; k < SIZES[l]; k++) row.push(rand() * 2 - 1);
      layer.push(row);
      biases.push(rand() * BIAS_MAX);
    }
    W.push(layer);
    b.push(biases);
  }
  return { W, b };
}
