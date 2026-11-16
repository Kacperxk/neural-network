import { describe, expect, it } from 'vitest';
import { allRefs, getParam } from '../src/engine/params';
import { initParams } from '../src/engine/rng';
import { traceBatch, traceSample } from '../src/engine/trace';
import { LAST_STAGE, reveal, STAGES, stepBack, stepNext, type Stepper } from '../src/model/stepper';
import { Store } from '../src/model/store';

const start = (): Stepper => ({ index: 0, iteration: 0, params: initParams(33), sample: 2, lr: 0.1 });

describe('step-through state machine', () => {
  it('has 13 stages and reveals things in a strictly growing order', () => {
    expect(STAGES).toHaveLength(13);
    let prev = reveal(0);
    expect(prev).toEqual({ input: false, fwd: 0, loss: false, deltaFrom: 3, gradFrom: 3, updated: false });
    for (let i = 1; i <= LAST_STAGE; i++) {
      const r = reveal(i);
      expect(r.fwd).toBeGreaterThanOrEqual(prev.fwd);
      expect(r.deltaFrom).toBeLessThanOrEqual(prev.deltaFrom);
      expect(r.gradFrom).toBeLessThanOrEqual(prev.gradFrom);
      // A gradient needs its layer's delta first; a delta needs the loss; the loss needs the full forward pass.
      expect(r.gradFrom).toBeGreaterThanOrEqual(r.deltaFrom);
      if (r.deltaFrom < 3) expect(r.loss).toBe(true);
      if (r.loss) expect(r.fwd).toBe(3);
      prev = r;
    }
    expect(reveal(LAST_STAGE)).toEqual({ input: true, fwd: 3, loss: true, deltaFrom: 0, gradFrom: 0, updated: true });
  });

  it('deltas are revealed output first, layer 1 last', () => {
    const order = STAGES.map((_, i) => reveal(i).deltaFrom);
    expect(order).toEqual([3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 0, 0, 0]);
  });

  it('Next walks every stage without touching the weights until the wrap', () => {
    let s = start();
    const p0 = s.params;
    for (let i = 1; i <= LAST_STAGE; i++) {
      s = stepNext(s);
      expect(s.index).toBe(i);
      expect(s.params).toBe(p0);
    }
    s = stepNext(s);
    expect(s.index).toBe(1);
    expect(s.iteration).toBe(1);
    expect(s.params).not.toBe(p0);
  });

  it('the committed update is exactly w - lr * dL/dw for the selected sample', () => {
    let s = start();
    const grads = traceSample(s.params, s.sample).grads;
    const before = s.params;
    for (let i = 0; i <= LAST_STAGE; i++) s = stepNext(s);
    for (const ref of allRefs()) {
      expect(getParam(s.params, ref)).toBe(getParam(before, ref) - 0.1 * getParam(grads, ref));
    }
  });

  it('Back then Next returns to the same state; Back at idle is a no-op', () => {
    let s = start();
    expect(stepBack(s)).toBe(s);
    for (let i = 0; i < 7; i++) s = stepNext(s);
    const again = stepNext(stepBack(s));
    expect(again).toEqual(s);
  });

  it('one single-sample update lowers that sample loss for the default preset at lr 0.1', () => {
    let s = start();
    const before = traceSample(s.params, s.sample).loss;
    for (let i = 0; i <= LAST_STAGE; i++) s = stepNext(s);
    expect(traceSample(s.params, s.sample).loss).toBeLessThan(before);
  });
});

describe('store', () => {
  it('train steps follow the mean gradient and record the loss history', () => {
    const store = new Store(true);
    store.setMode('train');
    const p0 = store.s.params;
    const b0 = traceBatch(p0);
    store.trainStep(1);
    for (const ref of allRefs()) {
      expect(getParam(store.s.params, ref)).toBe(getParam(p0, ref) - store.s.lr * getParam(b0.grads, ref));
    }
    expect(store.s.train.history).toHaveLength(2);
    expect(store.s.train.history[0]).toBe(b0.loss);
    expect(store.s.train.history[1]).toBe(traceBatch(store.s.params).loss);
    store.trainStep(5);
    expect(store.s.train.iter).toBe(6);
    expect(store.s.train.history).toHaveLength(7);
  });

  it('training the default preset reaches the documented loss', () => {
    const store = new Store(true);
    store.setMode('train');
    store.trainStep(133);
    expect(store.batch.loss).toBeLessThan(1e-3);
  });

  it('verify agrees with backprop on a weight', () => {
    const store = new Store(true);
    store.verify({ kind: 'w', l: 1, j: 0, k: 0 });
    const v = store.s.verify!;
    expect(v.analytic).toBe(store.snap.grads.W[1][0][0]);
    expect(v.relError).toBeLessThan(1e-6);
  });

  it('reset returns to the seed weights', () => {
    const store = new Store(true);
    const p0 = JSON.stringify(store.s.params);
    store.trainStep(10);
    store.resetWeights();
    expect(JSON.stringify(store.s.params)).toBe(p0);
    expect(store.s.train.history).toHaveLength(0);
  });
});
