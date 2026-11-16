import { describe, expect, it } from 'vitest';
import { N_LAYERS, SIZES, stepParams } from '../src/engine/params';
import { initParams } from '../src/engine/rng';
import { traceBatch, traceSample } from '../src/engine/trace';
import { fmt, fmtCode, fmtTex } from '../src/model/format';
import { reveal, REVEAL_ALL } from '../src/model/stepper';
import type { Selection } from '../src/model/store';
import { explain, type ExplainContext } from '../src/ui/explain';
import { tex } from '../src/ui/katex';

function everySelection(): Exclude<Selection, null>[] {
  const out: Exclude<Selection, null>[] = [];
  for (let col = 0; col < SIZES.length; col++) for (let i = 0; i < SIZES[col]; i++) out.push({ kind: 'neuron', col, i });
  for (let l = 0; l < N_LAYERS; l++) {
    for (let j = 0; j < SIZES[l + 1]; j++) {
      out.push({ kind: 'bias', l, j });
      for (let k = 0; k < SIZES[l]; k++) out.push({ kind: 'weight', l, j, k });
    }
  }
  return out;
}

function ctx(seed: number, sample: number, index: number | 'all', train = false): ExplainContext {
  const params = initParams(seed);
  const snap = traceSample(params, sample);
  const batch = traceBatch(params);
  return {
    snap,
    reveal: index === 'all' ? REVEAL_ALL : reveal(index),
    lr: 0.1,
    after: stepParams(params, train ? batch.grads : snap.grads, 0.1),
    batch: train ? { perSample: batch.samples.map((s) => s.grads), mean: batch.grads } : null,
  };
}

const strip = (html: string): string => html.replace(/<[^>]+>/g, '');

describe('inspector layers never contradict each other', () => {
  it('there are 34 inspectable objects: 9 neurons, 18 weights, 7 biases', () => {
    expect(everySelection()).toHaveLength(9 + 18 + 7);
  });

  for (const seed of [33, 122, 43, 235]) {
    for (let sample = 0; sample < 4; sample++) {
      it(`seed ${seed}, sample ${sample}: every shared number appears, identically rounded, in all three layers`, () => {
        for (const train of [false, true]) {
          const c = ctx(seed, sample, 'all', train);
          for (const sel of everySelection()) {
            const ex = explain(sel, c);
            const intuition = strip(ex.intuition);
            const math = ex.math.join('\n');
            expect(ex.shared.length, ex.title).toBeGreaterThan(0);
            for (const v of ex.shared) {
              expect(intuition, `${ex.title} intuition lacks ${fmt(v)}`).toContain(fmt(v));
              expect(math, `${ex.title} math lacks ${fmtTex(v)}`).toContain(fmtTex(v));
              expect(ex.code, `${ex.title} code lacks ${fmtCode(v)}`).toContain(fmtCode(v));
            }
            expect(intuition).not.toContain('?');
            expect(ex.code).not.toContain('?');
          }
        }
      });
    }
  }

  it('the gradient quoted on a weight card is the engine gradient, and the update is w - lr * g', () => {
    const c = ctx(33, 2, 'all');
    const ex = explain({ kind: 'weight', l: 1, j: 0, k: 0 }, c);
    const g = c.snap.grads.W[1][0][0];
    const w = c.snap.params.W[1][0][0];
    expect(ex.shared).toContain(g);
    expect(ex.shared).toContain(w - 0.1 * g);
    expect(c.snap.delta[1][0] * c.snap.a[1][0]).toBe(g);
  });

  it('nothing is quoted before the step-through has reached it', () => {
    for (let index = 0; index <= 12; index++) {
      const c = ctx(33, 2, index);
      for (const sel of everySelection()) {
        const ex = explain(sel, c);
        const all = strip(ex.intuition) + ex.math.join(' ') + ex.code;
        if (sel.kind === 'weight' && sel.l < c.reveal.gradFrom) {
          const g = c.snap.grads.W[sel.l][sel.j][sel.k];
          // A gradient that is exactly 0 formats like other zeros, so only non-zero ones can be checked.
          if (g !== 0 && Math.abs(g) > 0.0005) expect(ex.shared).not.toContain(g);
          expect(all).toContain('?');
        }
        if (sel.kind === 'neuron' && sel.col >= 1 && sel.col - 1 < c.reveal.deltaFrom) {
          expect(ex.code).toContain('?');
          expect(ex.math.join(' ')).toContain('\\text{?}');
        }
      }
    }
  });

  it('every formula is valid KaTeX', () => {
    for (const train of [false, true]) {
      for (const index of [0, 5, 'all'] as const) {
        const c = ctx(122, 1, index, train);
        for (const sel of everySelection()) {
          for (const m of explain(sel, c).math) {
            const html = tex(m);
            expect(html, m).not.toContain('katex-error');
          }
        }
      }
    }
  });
});
