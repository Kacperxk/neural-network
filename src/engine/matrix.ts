/**
 * The same network written the "numpy way": an explicit forward cache and a hand-derived backward.
 * Mirrors numpy_neural_net/model.py (see /reference). It exists for two reasons:
 *  1. it is the matrix-form derivation from the article, tested against the autograd engine;
 *  2. its plain-number forward() is cheap enough to fill the decision map and the loss surface.
 */
import { XOR } from './data';
import { N_LAYERS, SIZES, zerosLike, type Activation, type Params } from './params';

export type Cache = {
  /** a[0] = x */
  a: number[][];
  z: number[][];
  yhat: number;
};

function phi(z: number, act: Activation): number {
  return act === 'relu' ? Math.max(z, 0) : Math.tanh(z);
}

/** Same convention as the engine: the ReLU derivative at exactly z = 0 is 0. */
function dphi(z: number, act: Activation): number {
  if (act === 'relu') return z > 0 ? 1 : 0;
  const t = Math.tanh(z);
  return 1 - t * t;
}

/** z(l) = W(l) a(l-1) + b(l), a(l) = phi(z(l)); the last layer is linear. */
export function forward(p: Params, x: readonly number[], act: Activation = 'relu'): Cache {
  const a: number[][] = [x.slice()];
  const z: number[][] = [];
  for (let l = 0; l < N_LAYERS; l++) {
    const zl: number[] = [];
    const al: number[] = [];
    const isLast = l === N_LAYERS - 1;
    for (let j = 0; j < SIZES[l + 1]; j++) {
      // Same summation order as the engine (bias first), so results agree bit for bit.
      let s = p.b[l][j];
      for (let k = 0; k < SIZES[l]; k++) s += p.W[l][j][k] * a[l][k];
      zl.push(s);
      al.push(isLast ? s : phi(s, act));
    }
    z.push(zl);
    a.push(al);
  }
  return { a, z, yhat: a[N_LAYERS][0] };
}

export function predict(p: Params, x: readonly number[], act: Activation = 'relu'): number {
  return forward(p, x, act).yhat;
}

export function sampleLoss(p: Params, i: number, act: Activation = 'relu'): number {
  const d = predict(p, XOR[i].x, act) - XOR[i].y;
  return d * d;
}

export function batchLoss(p: Params, act: Activation = 'relu'): number {
  let s = 0;
  for (let i = 0; i < XOR.length; i++) s += sampleLoss(p, i, act);
  return s / XOR.length;
}

export type Backward = { delta: number[][]; grads: Params };

/**
 * delta_out = 2 (yhat - y)
 * delta(l)  = (W(l+1)^T delta(l+1)) * phi'(z(l))      elementwise
 * dL/dW(l)  = delta(l) a(l-1)^T,   dL/db(l) = delta(l)
 */
export function backward(p: Params, cache: Cache, y: number, act: Activation = 'relu'): Backward {
  const delta: number[][] = new Array(N_LAYERS);
  delta[N_LAYERS - 1] = [2 * (cache.yhat - y)];
  for (let l = N_LAYERS - 2; l >= 0; l--) {
    const d: number[] = [];
    for (let k = 0; k < SIZES[l + 1]; k++) {
      let back = 0;
      for (let j = 0; j < SIZES[l + 2]; j++) back += p.W[l + 1][j][k] * delta[l + 1][j];
      d.push(back * dphi(cache.z[l][k], act));
    }
    delta[l] = d;
  }
  const grads = zerosLike();
  for (let l = 0; l < N_LAYERS; l++) {
    for (let j = 0; j < SIZES[l + 1]; j++) {
      grads.b[l][j] = delta[l][j];
      for (let k = 0; k < SIZES[l]; k++) grads.W[l][j][k] = delta[l][j] * cache.a[l][k];
    }
  }
  return { delta, grads };
}

/** Mean of the four per-sample gradients. */
export function batchGrads(p: Params, act: Activation = 'relu'): { loss: number; grads: Params } {
  const grads = zerosLike();
  let loss = 0;
  for (const s of XOR) {
    const cache = forward(p, s.x, act);
    const g = backward(p, cache, s.y, act).grads;
    loss += (cache.yhat - s.y) ** 2 / XOR.length;
    for (let l = 0; l < N_LAYERS; l++) {
      for (let j = 0; j < SIZES[l + 1]; j++) {
        grads.b[l][j] += g.b[l][j] / XOR.length;
        for (let k = 0; k < SIZES[l]; k++) grads.W[l][j][k] += g.W[l][j][k] / XOR.length;
      }
    }
  }
  return { loss, grads };
}
