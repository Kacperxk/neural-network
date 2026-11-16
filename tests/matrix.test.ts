import { describe, expect, it } from 'vitest';
import { XOR } from '../src/engine/data';
import { backward, batchGrads, batchLoss, forward } from '../src/engine/matrix';
import { allRefs, getParam, meanParams, refKey } from '../src/engine/params';
import { initParams } from '../src/engine/rng';
import { traceBatch, traceSample } from '../src/engine/trace';

const seeds = [1, 2, 3, 7, 21, 42, 99, 123, 2024, 31337];

describe('hand-written matrix-form backward vs the autograd engine', () => {
  for (const seed of seeds) {
    it(`seed ${seed}: forward cache, deltas and all 25 gradients agree`, () => {
      const p = initParams(seed);
      for (let s = 0; s < 4; s++) {
        const snap = traceSample(p, s);
        const cache = forward(p, XOR[s].x);
        const back = backward(p, cache, XOR[s].y);

        expect(cache.yhat).toBe(snap.yhat);
        expect(cache.z).toEqual(snap.z);
        expect(cache.a).toEqual(snap.a);
        for (let l = 0; l < 3; l++) {
          for (let j = 0; j < back.delta[l].length; j++) {
            expect(back.delta[l][j]).toBeCloseTo(snap.delta[l][j], 12);
          }
        }
        let n = 0;
        for (const ref of allRefs()) {
          expect(getParam(back.grads, ref), refKey(ref)).toBeCloseTo(getParam(snap.grads, ref), 12);
          n++;
        }
        expect(n).toBe(25);
      }
    });
  }

  it('batch gradient = mean of the four per-sample gradients (engine, one shared graph)', () => {
    for (const seed of seeds) {
      const p = initParams(seed);
      const batch = traceBatch(p);
      const mean = meanParams(batch.samples.map((s) => s.grads));
      const viaMatrix = batchGrads(p);
      expect(batch.loss).toBeCloseTo(batchLoss(p), 14);
      expect(batch.loss).toBeCloseTo(batch.samples.reduce((a, s) => a + s.loss, 0) / 4, 14);
      for (const ref of allRefs()) {
        expect(getParam(batch.grads, ref), refKey(ref)).toBeCloseTo(getParam(mean, ref), 13);
        expect(getParam(batch.grads, ref), refKey(ref)).toBeCloseTo(getParam(viaMatrix.grads, ref), 13);
      }
    }
  });
});

describe('snapshot internals come straight from the engine and are self-consistent', () => {
  it('each activation gradient is the sum of its recorded += terms, one per consumer', () => {
    const p = initParams(7);
    for (let s = 0; s < 4; s++) {
      const snap = traceSample(p, s);
      for (let l = 0; l < 3; l++) {
        for (let k = 0; k < snap.a[l].length; k++) {
          const terms = snap.aTerms[l][k];
          expect(terms).toHaveLength(snap.delta[l].length);
          expect(new Set(terms.map((t) => t.j)).size).toBe(terms.length);
          for (const t of terms) expect(t.amount).toBe(t.w * t.delta);
          expect(terms.reduce((acc, t) => acc + t.amount, 0)).toBeCloseTo(snap.dA[l][k], 14);
        }
      }
    }
  });

  it('the sum hands delta to every product; the output delta is 2 (yhat - y)', () => {
    const p = initParams(7);
    const snap = traceSample(p, 2);
    expect(snap.delta[2][0]).toBe(2 * (snap.yhat - snap.y));
    expect(snap.dLossdDiff).toBe(2 * snap.diff);
    for (let l = 0; l < 3; l++) {
      for (let j = 0; j < snap.delta[l].length; j++) {
        for (const g of snap.dProd[l][j]) expect(g).toBe(snap.delta[l][j]);
        // `+ 0` folds -0 into 0: a closed ReLU gate times a negative gradient is -0 in IEEE arithmetic.
        if (l < 2) expect(snap.delta[l][j] + 0).toBe(snap.dAct[l][j] * snap.dA[l + 1][j] + 0);
      }
    }
  });
});
