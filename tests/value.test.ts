import { describe, expect, it } from 'vitest';
import { Value } from '../src/engine/value';

describe('Value: local rules', () => {
  it('add passes the gradient to both inputs unchanged', () => {
    const a = new Value(2);
    const b = new Value(-5);
    const out = a.add(b);
    out.backward();
    expect(out.data).toBe(-3);
    expect(a.grad).toBe(1);
    expect(b.grad).toBe(1);
  });

  it('mul hands each input the other input', () => {
    const a = new Value(2);
    const b = new Value(-5);
    const out = a.mul(b);
    out.backward();
    expect(a.grad).toBe(-5);
    expect(b.grad).toBe(2);
    expect(out.local).toEqual([-5, 2]);
  });

  it('pow uses k * x^(k-1)', () => {
    const a = new Value(3);
    a.pow(2).backward();
    expect(a.grad).toBe(6);
  });

  it('relu is a gate: 1 above zero, 0 at or below', () => {
    const cases: [number, number][] = [
      [1.5, 1],
      [-0.2, 0],
      [0, 0],
    ];
    for (const [x, g] of cases) {
      const a = new Value(x);
      a.relu().backward();
      expect(a.grad).toBe(g);
    }
  });

  it('sub and neg are built from add and mul, as in microtensor', () => {
    const a = new Value(4);
    const b = new Value(1.5);
    const out = a.sub(b);
    out.backward();
    expect(out.data).toBe(2.5);
    expect(a.grad).toBe(1);
    expect(b.grad).toBe(-1);
  });
});

describe('Value: accumulation', () => {
  it('a * a at a = 2 gives grad 4 (two += of 2)', () => {
    const a = new Value(2);
    a.mul(a).backward();
    expect(a.grad).toBe(4);
    expect(a.terms.map((t) => t.amount)).toEqual([2, 2]);
  });

  it('a value with several consumers sums their contributions', () => {
    const a = new Value(3);
    const out = a.mul(2).add(a.mul(5)).add(a.pow(2)); // 2a + 5a + a^2
    out.backward();
    expect(a.grad).toBe(2 + 5 + 6);
    expect(a.terms).toHaveLength(3);
    expect(a.terms.reduce((s, t) => s + t.amount, 0)).toBe(a.grad);
  });

  it('backward() is repeatable: it clears old gradients first', () => {
    const a = new Value(2);
    const out = a.mul(a);
    out.backward();
    out.backward();
    expect(a.grad).toBe(4);
  });

  it('processes every consumer before the value it consumes', () => {
    const a = new Value(1.2);
    const b = a.mul(3);
    const c = b.add(a);
    const d = c.mul(b);
    const order = d.backward();
    const pos = (v: Value): number => order.indexOf(v);
    expect(pos(d)).toBeLessThan(pos(c));
    expect(pos(c)).toBeLessThan(pos(b));
    expect(pos(b)).toBeLessThan(pos(a));
  });
});
