/**
 * Single app state. The scene and every panel render from here; nothing keeps its own copy of a number.
 * Parameters are treated as immutable: every change replaces the object, which is what the caches key on.
 */
import { XOR } from '../engine/data';
import { EPS, minHiddenAbsZ, numericGrad, relativeError, type Nudge } from '../engine/gradcheck';
import { batchGrads } from '../engine/matrix';
import { getParam, SIZES, stepParams, type ParamRef, type Params } from '../engine/params';
import { DEFAULT_PRESET, PRESETS, TARGET_LOSS } from '../engine/presets';
import { initParams } from '../engine/rng';
import { traceBatch, traceSample, type BatchTrace, type Snapshot } from '../engine/trace';
import { LAST_STAGE, reveal, REVEAL_ALL, stepBack, stepNext, type Reveal } from './stepper';

export type Selection =
  | null
  /** col 0 = inputs, 1 and 2 = hidden layers, 3 = output. */
  | { kind: 'neuron'; col: number; i: number }
  | { kind: 'weight'; l: number; j: number; k: number }
  | { kind: 'bias'; l: number; j: number };

export type Mode = 'step' | 'train';
export type View = 'lab' | 'exploded' | 'studio';

export type SeedStatus = { converges: boolean; steps: number; finalLoss: number; lr: number };

export type VerifyResult = {
  ref: ParamRef;
  scope: number | 'batch';
  nudge: Nudge;
  analytic: number;
  relError: number;
  /** Smallest |z| of a hidden neuron: if tiny, a ReLU may flip inside the nudge. */
  minAbsZ: number;
  /** Bumped on every run so views can replay the twitch animation. */
  nonce: number;
};

export type AppState = {
  params: Params;
  seed: number;
  presetName: string | null;
  seedStatus: SeedStatus;
  lr: number;
  sample: number;
  mode: Mode;
  view: View;
  stepIndex: number;
  iteration: number;
  autoplay: boolean;
  selection: Selection;
  /** Which neuron is opened in the Exploded view: weight-layer l (0..2) and index j. */
  exploded: { l: number; j: number; step: number };
  train: { running: boolean; iter: number; history: number[]; speed: number; diverged: boolean };
  descent: { a: ParamRef; b: ParamRef; big: boolean };
  verify: VerifyResult | null;
  showWeightLabels: boolean;
  legendOpen: boolean;
  reducedMotion: boolean;
};

export const LR_MIN = 0.01;
export const LR_MAX = 1.5;
/** Steps per second in Train mode. */
export const SPEEDS = [2, 15, 60, 300];

/** Does plain GD from this seed reach the target? Uses the plain-number path (tested equal to the engine). */
export function checkSeed(seed: number, lr = 0.1, maxSteps = 3000): SeedStatus {
  let p = initParams(seed);
  let loss = Infinity;
  for (let i = 0; i <= maxSteps; i++) {
    const g = batchGrads(p);
    loss = g.loss;
    if (!Number.isFinite(loss) || loss > 1e6) return { converges: false, steps: i, finalLoss: loss, lr };
    if (loss < TARGET_LOSS) return { converges: true, steps: i, finalLoss: loss, lr };
    p = stepParams(p, g.grads, lr);
  }
  return { converges: false, steps: maxSteps, finalLoss: loss, lr };
}

/** Number of consumers of neuron (l, j)'s output: the next layer's neurons, or the loss for the output neuron. */
export function consumersOf(l: number): number {
  return l < 2 ? SIZES[l + 2] : 1;
}

/** Exploded view: forward, one arrival per consumer, gate, sum, multiply. */
export function explodedSteps(l: number): number {
  return consumersOf(l) + 4;
}

type Listener = (s: AppState, prev: AppState) => void;

export class Store {
  s: AppState;
  private listeners: Listener[] = [];
  private snapCache: { params: Params; sample: number; snap: Snapshot } | null = null;
  private batchCache: { params: Params; batch: BatchTrace } | null = null;
  private afterCache: { params: Params; sample: number; lr: number; after: Params } | null = null;

  constructor(reducedMotion = false) {
    const p = DEFAULT_PRESET;
    this.s = {
      params: initParams(p.seed),
      seed: p.seed,
      presetName: p.name,
      seedStatus: { converges: true, steps: p.steps, finalLoss: TARGET_LOSS, lr: p.lr },
      lr: p.lr,
      sample: 2,
      mode: 'step',
      view: 'lab',
      stepIndex: 0,
      iteration: 0,
      autoplay: false,
      selection: null,
      exploded: { l: 0, j: 0, step: 0 },
      train: { running: false, iter: 0, history: [], speed: 1, diverged: false },
      descent: { a: { kind: 'w', l: 2, j: 0, k: 0 }, b: { kind: 'w', l: 2, j: 0, k: 1 }, big: false },
      verify: null,
      showWeightLabels: false,
      legendOpen: false,
      reducedMotion,
    };
  }

  subscribe(fn: Listener): void {
    this.listeners.push(fn);
  }

  set(patch: Partial<AppState>): void {
    const prev = this.s;
    this.s = { ...prev, ...patch };
    for (const fn of this.listeners) fn(this.s, prev);
  }

  // ---- derived, always from the engine -------------------------------------------------------

  /** Forward + backward for the selected sample at the current weights. */
  get snap(): Snapshot {
    const { params, sample } = this.s;
    const c = this.snapCache;
    if (c && c.params === params && c.sample === sample) return c.snap;
    const snap = traceSample(params, sample);
    this.snapCache = { params, sample, snap };
    return snap;
  }

  /** Full-batch loss and gradient, plus the four per-sample traces. */
  get batch(): BatchTrace {
    const { params } = this.s;
    if (this.batchCache && this.batchCache.params === params) return this.batchCache.batch;
    const batch = traceBatch(params);
    this.batchCache = { params, batch };
    return batch;
  }

  /** The gradient the current mode would step along. */
  get grads(): Params {
    return this.s.mode === 'train' ? this.batch.grads : this.snap.grads;
  }

  /** The loss the current mode is descending. */
  get loss(): number {
    return this.s.mode === 'train' ? this.batch.loss : this.snap.loss;
  }

  /** Weights after one update in the current mode. */
  get after(): Params {
    const { params, sample, lr, mode } = this.s;
    const key = mode === 'train' ? -1 : sample;
    const c = this.afterCache;
    if (c && c.params === params && c.sample === key && c.lr === lr) return c.after;
    const after = stepParams(params, this.grads, lr);
    this.afterCache = { params, sample: key, lr, after };
    return after;
  }

  /** What the stage is allowed to show right now. */
  get reveal(): Reveal {
    if (this.s.mode === 'train' || this.s.view === 'exploded') return REVEAL_ALL;
    return reveal(this.s.stepIndex);
  }

  // ---- step-through ---------------------------------------------------------------------------

  private stepper() {
    const { stepIndex, iteration, params, sample, lr } = this.s;
    return { index: stepIndex, iteration, params, sample, lr };
  }

  next(): void {
    if (this.s.view === 'exploded') return this.explodedNext();
    if (this.s.mode === 'train') return this.trainStep(1);
    const n = stepNext(this.stepper());
    this.set({ stepIndex: n.index, iteration: n.iteration, params: n.params, verify: null });
  }

  back(): void {
    if (this.s.view === 'exploded') return this.explodedBack();
    if (this.s.mode === 'train') return;
    const n = stepBack(this.stepper());
    this.set({ stepIndex: n.index, autoplay: false });
  }

  goto(index: number): void {
    if (this.s.mode !== 'step') this.setMode('step');
    if (this.s.view === 'exploded') this.setView('lab');
    this.set({ stepIndex: Math.max(0, Math.min(LAST_STAGE, index)), autoplay: false });
  }

  toggleAuto(): void {
    if (this.s.mode === 'train') return this.trainToggle();
    this.set({ autoplay: !this.s.autoplay });
  }

  setSample(sample: number): void {
    if (sample === this.s.sample) return;
    // In step mode a new sample means a new iteration from the top; elsewhere it is just a different view.
    const restart = this.s.mode === 'step' && this.s.view !== 'exploded';
    this.set({
      sample,
      verify: null,
      ...(restart ? { stepIndex: Math.min(this.s.stepIndex, 1), autoplay: false } : {}),
      exploded: { ...this.s.exploded, step: 0 },
    });
  }

  setLr(lr: number): void {
    this.set({ lr: Math.max(LR_MIN, Math.min(LR_MAX, Math.round(lr * 100) / 100)) });
  }

  // ---- weights --------------------------------------------------------------------------------

  private loadSeed(seed: number, presetName: string | null, lr: number, status: SeedStatus): void {
    this.set({
      params: initParams(seed),
      seed,
      presetName,
      seedStatus: status,
      lr,
      stepIndex: 0,
      iteration: 0,
      autoplay: false,
      verify: null,
      exploded: { ...this.s.exploded, step: 0 },
      train: { ...this.s.train, running: false, iter: 0, history: [], diverged: false },
    });
  }

  setPreset(i: number): void {
    const p = PRESETS[i];
    this.loadSeed(p.seed, p.name, p.lr, { converges: true, steps: p.steps, finalLoss: TARGET_LOSS, lr: p.lr });
  }

  /** A fresh random draw. It is checked, not filtered: a draw that gets stuck is shown as such. */
  randomise(seed = 1 + Math.floor(Math.random() * 99999)): void {
    this.loadSeed(seed, null, this.s.lr, checkSeed(seed));
  }

  /** Back to the initial weights of the current seed. */
  resetWeights(): void {
    this.loadSeed(this.s.seed, this.s.presetName, this.s.lr, this.s.seedStatus);
  }

  // ---- train mode -----------------------------------------------------------------------------

  setMode(mode: Mode): void {
    if (mode === this.s.mode) return;
    this.set({
      mode,
      autoplay: false,
      verify: null,
      stepIndex: 0,
      train: { ...this.s.train, running: false },
    });
  }

  /** n steps of full-batch gradient descent: w <- w - lr * (mean of the four per-sample gradients). */
  trainStep(n: number): void {
    let params = this.s.params;
    const history = this.s.train.history.slice();
    let iter = this.s.train.iter;
    let diverged = false;
    for (let i = 0; i < n; i++) {
      const batch = i === 0 ? this.batch : traceBatch(params);
      if (!Number.isFinite(batch.loss) || batch.loss > 1e9) {
        diverged = true;
        break;
      }
      if (history.length === 0) history.push(batch.loss);
      params = stepParams(params, batch.grads, this.s.lr);
      iter++;
      history.push(traceBatchLoss(params));
    }
    this.set({
      params,
      verify: null,
      train: { ...this.s.train, history, iter, diverged, running: diverged ? false : this.s.train.running },
    });
  }

  trainToggle(): void {
    if (this.s.mode !== 'train') this.setMode('train');
    this.set({ train: { ...this.s.train, running: !this.s.train.running && !this.s.train.diverged } });
  }

  setSpeed(speed: number): void {
    this.set({ train: { ...this.s.train, speed } });
  }

  // ---- views ----------------------------------------------------------------------------------

  setView(view: View): void {
    if (view === this.s.view) return;
    const patch: Partial<AppState> = { view, autoplay: false };
    if (view === 'exploded') {
      const sel = this.s.selection;
      // Open the selected neuron if there is one; inputs have no computation to open.
      if (sel && sel.kind === 'neuron' && sel.col >= 1) patch.exploded = { l: sel.col - 1, j: sel.i, step: 0 };
      else patch.exploded = { ...this.s.exploded, step: 0 };
      patch.train = { ...this.s.train, running: false };
    }
    this.set(patch);
  }

  explode(l: number, j: number): void {
    this.set({ view: 'exploded', autoplay: false, exploded: { l, j, step: 0 }, train: { ...this.s.train, running: false } });
  }

  explodedNext(): void {
    const e = this.s.exploded;
    const max = explodedSteps(e.l) - 1;
    this.set({ exploded: { ...e, step: e.step >= max ? 0 : e.step + 1 } });
  }

  explodedBack(): void {
    const e = this.s.exploded;
    this.set({ exploded: { ...e, step: Math.max(0, e.step - 1) }, autoplay: false });
  }

  select(selection: Selection): void {
    this.set({ selection, verify: null });
  }

  // ---- verify ---------------------------------------------------------------------------------

  /** Nudge one parameter by +eps and -eps and compare the slope with the backprop gradient. */
  verify(ref: ParamRef): void {
    const scope = this.s.mode === 'train' ? ('batch' as const) : this.s.sample;
    const nudge = numericGrad(this.s.params, ref, scope, EPS);
    const analytic = getParam(this.grads, ref);
    this.set({
      verify: {
        ref,
        scope,
        nudge,
        analytic,
        relError: relativeError(analytic, nudge.numeric),
        minAbsZ: minHiddenAbsZ(this.s.params, scope),
        nonce: (this.s.verify?.nonce ?? 0) + 1,
      },
    });
  }
}

function traceBatchLoss(p: Params): number {
  return traceBatch(p).loss;
}

export function sampleLabel(i: number): string {
  const s = XOR[i];
  return `(${s.x[0]}, ${s.x[1]}) → ${s.y}`;
}
