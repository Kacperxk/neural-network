/** Plain-number parameters of the 2 -> 3 -> 3 -> 1 network, and helpers to address them. */

export const SIZES = [2, 3, 3, 1] as const;
/** Number of weight layers (3). Layer index l = 0..2 maps to the article's layer l + 1. */
export const N_LAYERS = SIZES.length - 1;
export const N_PARAMS = 25;

export type Activation = 'relu' | 'tanh';

export type Params = {
  /** W[l][j][k]: weight from neuron k of layer l to neuron j of layer l + 1. */
  W: number[][][];
  /** b[l][j]: bias of neuron j of layer l + 1. */
  b: number[][];
};

export type ParamRef =
  | { kind: 'w'; l: number; j: number; k: number }
  | { kind: 'b'; l: number; j: number };

export function cloneParams(p: Params): Params {
  return { W: p.W.map((m) => m.map((r) => r.slice())), b: p.b.map((r) => r.slice()) };
}

export function zerosLike(): Params {
  const W: number[][][] = [];
  const b: number[][] = [];
  for (let l = 0; l < N_LAYERS; l++) {
    W.push(Array.from({ length: SIZES[l + 1] }, () => new Array<number>(SIZES[l]).fill(0)));
    b.push(new Array<number>(SIZES[l + 1]).fill(0));
  }
  return { W, b };
}

const REFS: ParamRef[] = [];

/** Every parameter in a fixed order: layer by layer, neuron by neuron, weights then bias. */
export function allRefs(): readonly ParamRef[] {
  if (REFS.length) return REFS;
  const refs = REFS;
  for (let l = 0; l < N_LAYERS; l++) {
    for (let j = 0; j < SIZES[l + 1]; j++) {
      for (let k = 0; k < SIZES[l]; k++) refs.push({ kind: 'w', l, j, k });
      refs.push({ kind: 'b', l, j });
    }
  }
  return refs;
}

export function getParam(p: Params, r: ParamRef): number {
  return r.kind === 'w' ? p.W[r.l][r.j][r.k] : p.b[r.l][r.j];
}

export function setParam(p: Params, r: ParamRef, v: number): void {
  if (r.kind === 'w') p.W[r.l][r.j][r.k] = v;
  else p.b[r.l][r.j] = v;
}

export function refKey(r: ParamRef): string {
  return r.kind === 'w' ? `w${r.l}_${r.j}_${r.k}` : `b${r.l}_${r.j}`;
}

export function sameRef(a: ParamRef, b: ParamRef): boolean {
  return refKey(a) === refKey(b);
}

/** p - lr * g, as a new object. */
export function stepParams(p: Params, g: Params, lr: number): Params {
  const out = cloneParams(p);
  for (const r of allRefs()) setParam(out, r, getParam(p, r) - lr * getParam(g, r));
  return out;
}

export function meanParams(list: Params[]): Params {
  const out = zerosLike();
  for (const r of allRefs()) {
    let s = 0;
    for (const p of list) s += getParam(p, r);
    setParam(out, r, s / list.length);
  }
  return out;
}
