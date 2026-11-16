import { describe, expect, it } from 'vitest';
import { EPS, minHiddenAbsZ, numericGrad, relativeError } from '../src/engine/gradcheck';
import { allRefs, getParam, N_PARAMS, refKey, stepParams } from '../src/engine/params';
import { PRESETS } from '../src/engine/presets';
import { initParams } from '../src/engine/rng';
import { traceBatch, traceSample } from '../src/engine/trace';

const TOL = 1e-6;
// A central difference is only valid if no ReLU flips inside the +-eps nudge.
const KINK_MARGIN = 1e-3;
// Below this, both numbers are rounding noise of the difference quotient (loss ~1, eps 1e-5).
const NOISE = 1e-9;

const seeds = [...new Set([...PRESETS.map((p) => p.seed), 11, 12, 13, 14, 15])];

describe('engine gradients vs central finite differences (eps = 1e-5)', () => {
  it('addresses exactly 25 parameters', () => {
    expect(allRefs()).toHaveLength(N_PARAMS);
    expect(new Set(allRefs().map(refKey)).size).toBe(N_PARAMS);
  });

  for (const seed of seeds) {
    const p = initParams(seed);
    for (let s = 0; s < 4; s++) {
      it(`seed ${seed}, sample ${s}: all 25 parameters within ${TOL}`, () => {
        expect(minHiddenAbsZ(p, s), 'a hidden z sits on the ReLU kink').toBeGreaterThan(KINK_MARGIN);
        const snap = traceSample(p, s);
        let checked = 0;
        for (const ref of allRefs()) {
          const analytic = getParam(snap.grads, ref);
          const { numeric } = numericGrad(p, ref, s, EPS);
          checked++;
          if (Math.abs(analytic) < NOISE && Math.abs(numeric) < NOISE) continue;
          expect(relativeError(analytic, numeric), refKey(ref)).toBeLessThan(TOL);
        }
        expect(checked).toBe(N_PARAMS);
      });
    }

    it(`seed ${seed}, batch loss: all 25 parameters within ${TOL}`, () => {
      const batch = traceBatch(p);
      for (const ref of allRefs()) {
        const analytic = getParam(batch.grads, ref);
        const { numeric } = numericGrad(p, ref, 'batch', EPS);
        if (Math.abs(analytic) < NOISE && Math.abs(numeric) < NOISE) continue;
        expect(relativeError(analytic, numeric), refKey(ref)).toBeLessThan(TOL);
      }
    });
  }

  it('the check has teeth: a gradient that is 0.1% off fails it', () => {
    const p = initParams(seeds[0]);
    const snap = traceSample(p, 1);
    const ref = allRefs().find((r) => Math.abs(getParam(snap.grads, r)) > 1e-3)!;
    const { numeric } = numericGrad(p, ref, 1, EPS);
    expect(relativeError(getParam(snap.grads, ref) * 1.001, numeric)).toBeGreaterThan(TOL);
  });
});

describe('gradients partway through training', () => {
  it('still match finite differences after 50 GD steps on every preset', () => {
    for (const preset of PRESETS) {
      let p = initParams(preset.seed);
      for (let i = 0; i < 50; i++) p = stepParams(p, traceBatch(p).grads, preset.lr);
      if (minHiddenAbsZ(p, 'batch') < KINK_MARGIN) continue;
      const batch = traceBatch(p);
      for (const ref of allRefs()) {
        const analytic = getParam(batch.grads, ref);
        const { numeric } = numericGrad(p, ref, 'batch', EPS);
        if (Math.abs(analytic) < NOISE && Math.abs(numeric) < NOISE) continue;
        expect(relativeError(analytic, numeric), `${preset.name} ${refKey(ref)}`).toBeLessThan(TOL);
      }
    }
  });
});
