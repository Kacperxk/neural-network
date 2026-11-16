/**
 * Central finite differences, as in numpy_neural_net's gradient_check:
 * nudge one parameter by +eps and -eps, re-run the forward pass, and take the slope.
 */
import { XOR } from './data';
import { batchLoss, forward, sampleLoss } from './matrix';
import { cloneParams, getParam, setParam, N_LAYERS, type Activation, type ParamRef, type Params } from './params';

export const EPS = 1e-5;

export type Nudge = {
  eps: number;
  lossPlus: number;
  lossMinus: number;
  numeric: number;
};

/** `sample` = 0..3 for one sample's loss, or 'batch' for the mean over all four. */
export function numericGrad(
  p: Params,
  ref: ParamRef,
  sample: number | 'batch',
  eps = EPS,
  act: Activation = 'relu',
): Nudge {
  const q = cloneParams(p);
  const original = getParam(p, ref);
  const loss = (): number => (sample === 'batch' ? batchLoss(q, act) : sampleLoss(q, sample, act));
  setParam(q, ref, original + eps);
  const lossPlus = loss();
  setParam(q, ref, original - eps);
  const lossMinus = loss();
  return { eps, lossPlus, lossMinus, numeric: (lossPlus - lossMinus) / (2 * eps) };
}

/** |a - n| / (|a| + |n|), and 0 when both are exactly 0. */
export function relativeError(analytic: number, numeric: number): number {
  const denom = Math.abs(analytic) + Math.abs(numeric);
  return denom === 0 ? 0 : Math.abs(analytic - numeric) / denom;
}

/**
 * Smallest |z| among hidden neurons for the samples involved.
 * A finite difference is only meaningful if no ReLU switches on or off inside the +-eps nudge,
 * so callers warn when this is tiny.
 */
export function minHiddenAbsZ(p: Params, sample: number | 'batch', act: Activation = 'relu'): number {
  const idx = sample === 'batch' ? XOR.map((_, i) => i) : [sample];
  let min = Infinity;
  for (const i of idx) {
    const { z } = forward(p, XOR[i].x, act);
    for (let l = 0; l < N_LAYERS - 1; l++) for (const v of z[l]) min = Math.min(min, Math.abs(v));
  }
  return min;
}
