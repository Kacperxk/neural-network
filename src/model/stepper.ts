/**
 * The step-through state machine: one training iteration on one sample, as 13 explicit stages.
 * It is pure. The numbers come from a Snapshot taken once per iteration; the stepper only decides
 * how much of that snapshot is revealed, which is what makes Back trivial.
 */
import { stepParams, type Params } from '../engine/params';
import { traceSample } from '../engine/trace';

export const STAGES = [
  'idle',
  'input',
  'fwd1',
  'fwd2',
  'fwd3',
  'loss',
  'delta3',
  'grad3',
  'delta2',
  'grad2',
  'delta1',
  'grad1',
  'update',
] as const;

export type StageId = (typeof STAGES)[number];
export const LAST_STAGE = STAGES.length - 1;

export type Phase = 'idle' | 'forward' | 'loss' | 'backward' | 'update';

export function phaseOf(index: number): Phase {
  if (index === 0) return 'idle';
  if (index <= 4) return 'forward';
  if (index === 5) return 'loss';
  if (index <= 11) return 'backward';
  return 'update';
}

export const STAGE_TITLES: Record<StageId, string> = {
  idle: 'Ready',
  input: 'Load the sample',
  fwd1: 'Forward: layer 1',
  fwd2: 'Forward: layer 2',
  fwd3: 'Forward: output',
  loss: 'Loss',
  delta3: 'Backward: output δ',
  grad3: 'Backward: layer 3 gradients',
  delta2: 'Backward: layer 2 δ',
  grad2: 'Backward: layer 2 gradients',
  delta1: 'Backward: layer 1 δ',
  grad1: 'Backward: layer 1 gradients',
  update: 'Update',
};

export type Reveal = {
  /** x and y are on the table. */
  input: boolean;
  /** Number of weight layers whose z and a are known (0..3). */
  fwd: number;
  loss: boolean;
  /** delta of weight-layer l is known when l >= deltaFrom (3 = none yet). */
  deltaFrom: number;
  /** parameter gradients of weight-layer l are known when l >= gradFrom (3 = none yet). */
  gradFrom: number;
  /** The weights are drawn at their post-update values. */
  updated: boolean;
};

export function reveal(index: number): Reveal {
  return {
    input: index >= 1,
    fwd: Math.max(0, Math.min(3, index - 1)),
    loss: index >= 5,
    deltaFrom: index >= 10 ? 0 : index >= 8 ? 1 : index >= 6 ? 2 : 3,
    gradFrom: index >= 11 ? 0 : index >= 9 ? 1 : index >= 7 ? 2 : 3,
    updated: index >= 12,
  };
}

/** Everything is visible: used by Train mode and the Exploded view. */
export const REVEAL_ALL: Reveal = { input: true, fwd: 3, loss: true, deltaFrom: 0, gradFrom: 0, updated: false };

export type Stepper = {
  index: number;
  iteration: number;
  params: Params;
  sample: number;
  lr: number;
};

/** The parameters after this iteration's update: w - lr * dL/dw for this one sample. */
export function afterUpdate(s: Stepper): Params {
  return stepParams(s.params, traceSample(s.params, s.sample).grads, s.lr);
}

/** Next stage. From `update` it commits the new weights and starts the next iteration at `input`. */
export function stepNext(s: Stepper): Stepper {
  if (s.index < LAST_STAGE) return { ...s, index: s.index + 1 };
  return { ...s, index: 1, iteration: s.iteration + 1, params: afterUpdate(s) };
}

/** Previous stage. The weights are only committed on leaving `update` forwards, so Back never has to undo them. */
export function stepBack(s: Stepper): Stepper {
  return s.index > 0 ? { ...s, index: s.index - 1 } : s;
}

export function canBack(s: Stepper): boolean {
  return s.index > 0;
}
