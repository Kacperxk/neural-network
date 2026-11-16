import { Vector3 } from 'three';
import { SIZES } from '../engine/params';

/** One colour language for the scene, the formulas and the charts. Mirrored as CSS variables in styles.css. */
export const COLORS = {
  bg: 0x0b0f1a,
  /** forward values, activations */
  fwd: 0x5fd4ff,
  /** backward signal, deltas, gradients */
  back: 0xffb454,
  /** positive weight */
  pos: 0x4fe0b0,
  /** negative weight */
  neg: 0xff6b8a,
  /** target and loss */
  target: 0xb79cff,
  ink: 0xc3cadc,
  metal: 0x1a2133,
} as const;

export const HEX = {
  fwd: '#5fd4ff',
  back: '#ffb454',
  pos: '#4fe0b0',
  neg: '#ff6b8a',
  target: '#b79cff',
  ink: '#c3cadc',
} as const;

// ---- network layout ---------------------------------------------------------------------------

export const COL_X = [-4.2, -1.4, 1.4, 4.2];
export const NET_Y = 3.45;
export const ROW_GAP = 1.8;
export const NEURON_R = 0.4;
export const BENCH_TOP = 0.5;

/** Centre of neuron i in column col (0 = inputs ... 3 = output). Index 0 is the top neuron. */
export function neuronPos(col: number, i: number): Vector3 {
  const n = SIZES[col];
  return new Vector3(COL_X[col], NET_Y + ((n - 1) / 2 - i) * ROW_GAP, 0);
}

/** Tube radius for a weight. Square-root map, clamped: a weight of 2 or more draws at full thickness. */
export function tubeRadius(w: number): number {
  return 0.012 + 0.078 * Math.min(1, Math.sqrt(Math.abs(w) / 2));
}

export function biasRadius(b: number): number {
  return 0.05 + 0.09 * Math.min(1, Math.sqrt(Math.abs(b) / 2));
}

// ---- the loss gauge next to the output neuron -------------------------------------------------

export const GAUGE_X = COL_X[3] + 1.55;
/** World units per 1.0 of output value. */
export const GAUGE_SCALE = 1.25;
export const GAUGE_MIN = -1.2;
export const GAUGE_MAX = 2.2;

/** Height of an output value on the gauge. 0.5 sits level with the output neuron. */
export function gaugeY(v: number): number {
  const c = Math.max(GAUGE_MIN, Math.min(GAUGE_MAX, v));
  return NET_Y + (c - 0.5) * GAUGE_SCALE;
}

export const ease = (t: number): number => {
  const x = Math.max(0, Math.min(1, t));
  return x * x * (3 - 2 * x);
};
