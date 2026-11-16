import { describe, expect, it } from 'vitest';
import { XOR } from '../src/engine/data';
import { minHiddenAbsZ } from '../src/engine/gradcheck';
import { predict } from '../src/engine/matrix';
import { stepParams } from '../src/engine/params';
import { PRESETS, TARGET_LOSS } from '../src/engine/presets';
import { initParams } from '../src/engine/rng';
import { traceBatch } from '../src/engine/trace';

describe('shipped presets', () => {
  for (const preset of PRESETS) {
    it(`"${preset.name}" (seed ${preset.seed}) reaches loss < ${TARGET_LOSS} in exactly ${preset.steps} engine GD steps`, () => {
      let p = initParams(preset.seed);
      let steps = 0;
      for (; steps <= 3000; steps++) {
        const batch = traceBatch(p);
        if (batch.loss < TARGET_LOSS) break;
        p = stepParams(p, batch.grads, preset.lr);
      }
      expect(steps).toBe(preset.steps);
      for (const s of XOR) expect(Math.abs(predict(p, s.x) - s.y)).toBeLessThan(0.06);
    });

    it(`"${preset.name}" starts away from every ReLU kink, so finite differences are valid`, () => {
      expect(minHiddenAbsZ(initParams(preset.seed), 'batch')).toBeGreaterThan(0.02);
    });
  }
});
