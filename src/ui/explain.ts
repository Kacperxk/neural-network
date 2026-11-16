/**
 * Builds the three layers of an inspector card (Intuition / Math / Code) from ONE snapshot.
 * Every layer formats the same numbers with the same rounding, so they cannot contradict each other;
 * tests/explain.test.ts checks that. Values the step-through has not reached yet are shown as "?".
 */
import { XOR } from '../engine/data';
import type { ParamRef, Params } from '../engine/params';
import type { Snapshot } from '../engine/trace';
import { fmt, fmtCode, fmtTex } from '../model/format';
import type { Reveal } from '../model/stepper';
import type { Selection } from '../model/store';

export type ExplainContext = {
  snap: Snapshot;
  reveal: Reveal;
  lr: number;
  /** Parameters after the update the current mode would make. */
  after: Params;
  /** In Train mode: the four per-sample gradients and their mean. */
  batch: { perSample: Params[]; mean: Params } | null;
};

export type Explanation = {
  title: string;
  subtitle: string;
  /** HTML. */
  intuition: string;
  /** Display-mode TeX, one entry per line. */
  math: string[];
  /** Plain text (Python with comments). */
  code: string;
  /** Numbers that all three layers quote. Empty entries are not yet revealed. */
  shared: number[];
  param: ParamRef | null;
  /** Weight-layer and index of the neuron this card can open in the Exploded view. */
  neuron: { l: number; j: number } | null;
};

type Num = { t: string; x: string; c: string; known: boolean; v: number };

const num = (v: number, known = true): Num =>
  known ? { t: fmt(v), x: fmtTex(v), c: fmtCode(v), known, v } : { t: '?', x: '\\text{?}', c: '?', known, v };

const paren = (n: Num): string => `(${n.x})`;
const b = (n: Num): string => `<b>${n.t}</b>`;

// TeX symbols, coloured like the scene: forward values cyan, backward amber, weights green, target violet.
type Ix = number | string;
const A = (L: number, k: Ix): string => (L === 0 ? `\\fw{x_{${k}}}` : `\\fw{a^{(${L})}_{${k}}}`);
const Z = (L: number, j: Ix): string => (L === 3 ? `z^{(3)}` : `z^{(${L})}_{${j}}`);
const D = (L: number, j: Ix): string => (L === 3 ? `\\bk{\\delta^{(3)}}` : `\\bk{\\delta^{(${L})}_{${j}}}`);
const W = (L: number, j: Ix, k: Ix): string => `\\wt{w^{(${L})}_{${j}${k}}}`;
const B = (L: number, j: Ix): string => `\\wt{b^{(${L})}_{${j}}}`;

function srcName(l: number, k: number): string {
  return l === 0 ? `input x${sub(k + 1)}` : `hidden ${l}, neuron ${k + 1}`;
}
function dstName(l: number, j: number): string {
  return l === 2 ? 'the output neuron' : `hidden ${l + 1}, neuron ${j + 1}`;
}
function sub(n: number): string {
  return `<sub>${n}</sub>`;
}

const PY_NOTE = '# Python counts from 0, the formulas from 1.';

function signSentence(g: number, a: number, delta: number, what: string): string {
  if (g > 0) return `It is positive: making this ${what} larger would make the loss larger, so the update makes it smaller.`;
  if (g < 0) return `It is negative: making this ${what} larger would make the loss smaller, so the update makes it larger.`;
  if (a === 0 && what === 'weight') return `It is exactly 0 because the signal it carried was 0. A weight that passed nothing on cannot be blamed for this point.`;
  if (delta === 0) return `It is exactly 0 because the neuron it feeds has δ = 0: no change to this ${what} can reach the loss on this point.`;
  return `It is exactly 0.`;
}

function batchBlock(ctx: ExplainContext, pick: (p: Params) => number, sym: string, what: string): { i: string; m: string[]; c: string; shared: number[] } {
  if (!ctx.batch) return { i: '', m: [], c: '', shared: [] };
  const gs = ctx.batch.perSample.map(pick);
  const mean = pick(ctx.batch.mean);
  const t = gs.map((g) => fmt(g));
  return {
    i: `<p>Training looks at all four points. Their gradients for this ${what} are ${t.join(', ')}; the mean is <b>${fmt(mean)}</b>, and that mean is what the step uses.</p>`,
    m: [
      `\\frac{\\partial \\tg{\\bar L}}{\\partial ${sym}} = \\frac14\\sum_{n=1}^{4}\\frac{\\partial \\tg{L_n}}{\\partial ${sym}} = \\frac14\\big(${gs.map((g) => `(${fmtTex(g)})`).join('+')}\\big) = \\bk{${fmtTex(mean)}}`,
    ],
    c: `\n# full batch: one graph holding four copies of the network\nloss = sum(losses) * (1 / 4)\n# p.grad gets one += per point:\n#   (${gs.map((g) => fmtCode(g)).join(' + ')}) / 4 = ${fmtCode(mean)}\n`,
    shared: [mean],
  };
}

function explainWeight(ctx: ExplainContext, l: number, j: number, k: number): Explanation {
  const { snap, reveal: r, lr } = ctx;
  const L = l + 1;
  const aKnown = l === 0 ? r.input : r.fwd >= l;
  const gKnown = l >= r.gradFrom;
  const w = num(snap.params.W[l][j][k]);
  const a = num(snap.a[l][k], aKnown);
  const p = num(snap.prod[l][j][k], aKnown);
  const d = num(snap.delta[l][j], gKnown);
  const g = num(snap.grads.W[l][j][k], gKnown);
  const batch = batchBlock(ctx, (q) => q.W[l][j][k], W(L, j + 1, k + 1), 'weight');
  const stepG = ctx.batch ? num(ctx.batch.mean.W[l][j][k]) : g;
  const w2 = num(ctx.after.W[l][j][k], stepG.known);
  const eta = num(lr);
  const aSym = l === 0 ? 'x' : 'a';

  const intuition =
    `<p>This tube carries the signal from ${srcName(l, k)} to ${dstName(l, j)}, scaled by the weight <b>w = ${w.t}</b>. ` +
    (aKnown ? `Right now it carries ${aSym} × w = ${a.t} × ${w.t} = ${b(p)}.` : `Run the forward pass to see what it carries.`) +
    `</p>` +
    (gKnown
      ? `<p>The neuron it feeds has error signal δ = ${b(d)}: how much the loss changes per unit of that neuron's sum z. This weight added ${aSym} × w to that sum, so its share of the blame is δ × ${aSym} = ${d.t} × ${a.t} = ${b(g)}. ${signSentence(g.v, a.v, d.v, 'weight')}</p>`
      : `<p>Its gradient is not known yet: step backward until this layer's δ exists.</p>`) +
    batch.i +
    (stepG.known ? `<p>Update with η = ${eta.t}: w − η × gradient = ${w.t} − ${eta.t} × ${stepG.t} = ${b(w2)}.</p>` : '');

  const math = [
    `${Z(L, j + 1)} = \\sum_k ${W(L, j + 1, 'k')}\\,${A(l, 'k')} + ${B(L, j + 1)}`,
    `${W(L, j + 1, k + 1)}\\,${A(l, k + 1)} = ${paren(w)}${paren(a)} = ${p.x}`,
    `\\frac{\\partial \\tg{L}}{\\partial ${W(L, j + 1, k + 1)}} = ${D(L, j + 1)}\\,${A(l, k + 1)} = ${paren(d)}${paren(a)} = \\bk{${g.x}}`,
    ...batch.m,
    `${W(L, j + 1, k + 1)} \\leftarrow w - \\eta\\,\\frac{\\partial \\tg{L}}{\\partial w} = ${w.x} - ${paren(eta)}${paren(stepG)} = \\wt{${w2.x}}`,
  ];

  const code =
    `${PY_NOTE}\n# nn.py · Neuron.__call__\n` +
    `val += self.w[${k}] * x[${k}]      # ${w.c} * ${a.c} = ${p.c}\n\n` +
    `# engine.py · Value.__mul__  (self = w, other = x)\n` +
    `self.grad += other.data * out.grad\n` +
    `#          += ${a.c} * ${d.c} = ${g.c}\n` +
    `# out.grad is δ here: every + on the way\n# passed it along unchanged.\n` +
    batch.c +
    `\n# the update\np.data -= lr * p.grad\n#   ${w.c} - ${eta.c} * ${stepG.c} = ${w2.c}`;

  return {
    title: `Weight w${L} · ${l === 0 ? 'x' : 'h' + l}${k + 1} → ${l === 2 ? 'ŷ' : 'h' + L + (j + 1)}`,
    subtitle: `From ${srcName(l, k).replace(/<[^>]+>/g, '')} to ${dstName(l, j)}`,
    intuition,
    math,
    code,
    shared: [w.v, ...(aKnown ? [a.v, p.v] : []), ...(gKnown ? [d.v, g.v] : []), ...(stepG.known ? [w2.v] : []), ...batch.shared],
    param: { kind: 'w', l, j, k },
    neuron: null,
  };
}

function explainBias(ctx: ExplainContext, l: number, j: number): Explanation {
  const { snap, reveal: r, lr } = ctx;
  const L = l + 1;
  const gKnown = l >= r.gradFrom;
  const bv = num(snap.params.b[l][j]);
  const d = num(snap.delta[l][j], gKnown);
  const g = num(snap.grads.b[l][j], gKnown);
  const batch = batchBlock(ctx, (q) => q.b[l][j], B(L, j + 1), 'bias');
  const stepG = ctx.batch ? num(ctx.batch.mean.b[l][j]) : g;
  const b2 = num(ctx.after.b[l][j], stepG.known);
  const eta = num(lr);

  const intuition =
    `<p>The bias is a number added straight into the sum of ${dstName(l, j)}: z = (weighted inputs) + b, with <b>b = ${bv.t}</b>. It lets the neuron switch on or off at a point other than zero.</p>` +
    (gKnown
      ? `<p>Because it is added, not multiplied, nudging b moves z by exactly the same amount. So its gradient is the neuron's δ itself: 1 × ${d.t} = ${b(g)}. ${signSentence(g.v, 1, d.v, 'bias')}</p>`
      : `<p>Its gradient is not known yet: step backward until this layer's δ exists.</p>`) +
    batch.i +
    (stepG.known ? `<p>Update with η = ${eta.t}: b − η × gradient = ${bv.t} − ${eta.t} × ${stepG.t} = ${b(b2)}.</p>` : '');

  const math = [
    `${Z(L, j + 1)} = \\sum_k ${W(L, j + 1, 'k')}\\,${A(l, 'k')} + ${B(L, j + 1)}, \\qquad ${B(L, j + 1)} = ${bv.x}`,
    `\\frac{\\partial \\tg{L}}{\\partial ${B(L, j + 1)}} = ${D(L, j + 1)}\\cdot 1 = ${paren(d)}(1) = \\bk{${g.x}}`,
    ...batch.m,
    `${B(L, j + 1)} \\leftarrow b - \\eta\\,\\frac{\\partial \\tg{L}}{\\partial b} = ${bv.x} - ${paren(eta)}${paren(stepG)} = \\wt{${b2.x}}`,
  ];

  const code =
    `# nn.py · Neuron.__call__\nval = self.b                    # ${bv.c}\nval += self.w[i] * x[i]         # then each input\n\n` +
    `# engine.py · Value.__add__  (self = b)\nself.grad += 1.0 * out.grad\n#          += 1.0 * ${d.c} = ${g.c}\n` +
    batch.c +
    `\n# the update\np.data -= lr * p.grad\n#   ${bv.c} - ${eta.c} * ${stepG.c} = ${b2.c}`;

  return {
    title: `Bias b${L} · ${l === 2 ? 'ŷ' : 'h' + L + (j + 1)}`,
    subtitle: `Added to the sum of ${dstName(l, j)}`,
    intuition,
    math,
    code,
    shared: [bv.v, ...(gKnown ? [d.v, g.v] : []), ...(stepG.known ? [b2.v] : [])],
    param: { kind: 'b', l, j },
    neuron: null,
  };
}

function explainHidden(ctx: ExplainContext, col: number, j: number): Explanation {
  const { snap, reveal: r } = ctx;
  const l = col - 1;
  const L = col;
  const inKnown = l === 0 ? r.input : r.fwd >= l;
  const zKnown = r.fwd >= col;
  const dKnown = l >= r.deltaFrom;
  const ws = snap.params.W[l][j];
  const as = snap.a[l];
  const bias = snap.params.b[l][j];
  const z = num(snap.z[l][j], zKnown);
  const a = num(snap.a[col][j], zKnown);
  const gate = snap.dAct[l][j];
  const terms = snap.aTerms[l + 1][j].slice().sort((p, q) => p.j - q.j);
  const dA = num(snap.dA[col][j], dKnown);
  const d = num(snap.delta[l][j], dKnown);
  const n = terms.length;
  const next = l === 1 ? 'the output neuron' : `the ${n} neurons of hidden ${col + 1}`;
  const off = zKnown && gate === 0;

  const intuition =
    `<p>` +
    (zKnown
      ? `It multiplies each input by its weight, adds its bias, and gets z = ${b(z)}. ReLU keeps positive numbers and turns everything else into 0, so its output is a = ${b(a)}.${off ? ' For this point the neuron is switched off.' : ''}`
      : `It will multiply each input by its weight, add its bias to get z, and pass z through ReLU to get its output a. Step forward to see the numbers.`) +
    `</p>` +
    (dKnown
      ? `<p>Going backward, ${next} ${n === 1 ? 'sends' : 'each send'} back ${n === 1 ? 'its' : 'their own'} δ times the connecting weight: ${terms
          .map((t) => fmt(t.amount))
          .join(', ')}.${n > 1 ? ` Added together that is ${b(dA)}:` : ` That is ${b(dA)}:`} how much the loss cares about this neuron's output. The ReLU gate is ${
          gate === 1 ? 'open (× 1)' : 'closed (× 0)'
        }, so δ = ${gate} × ${dA.t} = ${b(d)}.${gate === 0 ? ' Nothing flows back through a closed gate, so the weights into this neuron get a gradient of 0 for this point.' : ''}</p>`
      : `<p>Its δ is not known yet. It needs the δ of every neuron it feeds first, which is why the backward pass starts at the output.</p>`);

  const math = [
    `${Z(L, j + 1)} = \\sum_k ${W(L, j + 1, 'k')}\\,${A(l, 'k')} + ${B(L, j + 1)} = ${ws
      .map((w, k) => `(${fmtTex(w)})${paren(num(as[k], inKnown))}`)
      .join('+')}+(${fmtTex(bias)}) = ${z.x}`,
    `${A(L, j + 1)} = \\max(0,\\,${Z(L, j + 1)}) = \\fw{${a.x}}`,
    `${D(L, j + 1)} = \\Big(\\sum_i ${W(L + 1, 'i', j + 1)}\\,${l === 1 ? D(3, 1) : `\\bk{\\delta^{(${L + 1})}_{i}}`}\\Big)\\,\\varphi'(${Z(L, j + 1)})`,
    `= \\big(${terms.map((t) => `(${fmtTex(t.w)})${paren(num(t.delta, dKnown))}`).join('+')}\\big)\\,(${zKnown ? gate : '\\text{?}'}) = ${paren(dA)}(${zKnown ? gate : '\\text{?}'}) = \\bk{${d.x}}`,
  ];

  const code =
    `# nn.py · Neuron.__call__\nval = self.b                 # ${fmtCode(bias)}\n` +
    ws.map((w, k) => `val += self.w[${k}] * x[${k}]   # ${fmtCode(w)} * ${num(as[k], inKnown).c}\n`).join('') +
    `return val.relu()            # max(${z.c}, 0) = ${a.c}\n\n` +
    `# engine.py · Value.__mul__ of each consumer\n# adds one term to this neuron's output:\nself.grad += other.data * out.grad\n` +
    terms.map((t) => `#   += ${fmtCode(t.w)} * ${num(t.delta, dKnown).c} = ${num(t.amount, dKnown).c}\n`).join('') +
    `#   total ${dA.c}\n\n` +
    `# engine.py · Value.relu  (self = z)\nself.grad += (1 if self.data > 0 else 0) * out.grad\n#          += ${zKnown ? gate : '?'} * ${dA.c} = ${d.c}`;

  return {
    title: `Hidden ${col} · neuron ${j + 1}`,
    subtitle: 'Weighted sum, then ReLU',
    intuition,
    math,
    code,
    shared: [...(zKnown ? [z.v, a.v] : []), ...(dKnown ? [dA.v, d.v] : [])],
    param: null,
    neuron: { l, j },
  };
}

function explainOutput(ctx: ExplainContext): Explanation {
  const { snap, reveal: r } = ctx;
  const known = r.fwd >= 3;
  const lossKnown = r.loss;
  const dKnown = 2 >= r.deltaFrom;
  const ws = snap.params.W[2][0];
  const as = snap.a[2];
  const bias = snap.params.b[2][0];
  const yhat = num(snap.yhat, known);
  const y = num(snap.y, r.input);
  const diff = num(snap.diff, lossKnown);
  const loss = num(snap.loss, lossKnown);
  const d = num(snap.delta[2][0], dKnown);

  const intuition =
    `<p>The output neuron has no ReLU. It adds up its three weighted inputs and its bias, and that sum is the network's answer` +
    (known ? `: ŷ = ${b(yhat)}.` : `. Step forward to see it.`) +
    `</p>` +
    (lossKnown ? `<p>The target is y = ${y.t}, so the gap is ŷ − y = ${b(diff)}. Squaring it gives the loss L = ${b(loss)}: always positive, and big misses count much more than small ones.</p>` : '') +
    (dKnown
      ? `<p>The slope of a square is twice its side, so δ = 2 × gap = ${b(d)}. ${
          d.v > 0 ? 'Positive: the answer is too high, and raising it further would make the loss worse.' : d.v < 0 ? 'Negative: the answer is too low, and raising it would shrink the loss.' : 'Zero: the answer is exactly right.'
        } Every other gradient in the network is built from this one number.</p>`
      : lossKnown
        ? `<p>Step once more to start the backward pass here.</p>`
        : '');

  const math = [
    `\\fw{\\hat y} = z^{(3)} = ${ws.map((w, k) => `(${fmtTex(w)})${paren(num(as[k], r.fwd >= 2))}`).join('+')}+(${fmtTex(bias)}) = \\fw{${yhat.x}}`,
    `\\tg{L} = (\\fw{\\hat y} - \\tg{y})^2 = (${yhat.x} - ${y.x})^2 = (${diff.x})^2 = \\tg{${loss.x}}`,
    `${D(3, 1)} = \\frac{\\partial \\tg{L}}{\\partial \\hat y} = 2(\\fw{\\hat y} - \\tg{y}) = 2(${diff.x}) = \\bk{${d.x}}`,
  ];

  const code =
    `# nn.py · last layer: nonlin=False, no relu\nyhat = model(x)               # ${yhat.c}\nloss = (yhat - y) ** 2        # (${yhat.c} - ${y.c}) ** 2 = ${loss.c}\n\n` +
    `# engine.py · Value.__pow__  (self = yhat - y, other = 2)\nself.grad += other * self.data ** (other - 1) * out.grad\n#          += 2 * ${diff.c} * 1.0 = ${d.c}\n` +
    `# out.grad is 1.0: backward() starts by\n# setting loss.grad = 1.0`;

  return {
    title: 'Output neuron ŷ',
    subtitle: 'Linear: a weighted sum with no ReLU',
    intuition,
    math,
    code,
    shared: [...(known ? [yhat.v] : []), ...(lossKnown ? [loss.v] : []), ...(dKnown ? [d.v] : [])],
    param: null,
    neuron: { l: 2, j: 0 },
  };
}

function explainInput(ctx: ExplainContext, i: number): Explanation {
  const { snap, reveal: r } = ctx;
  const known = r.input;
  const gKnown = r.deltaFrom === 0;
  const x = num(snap.x[i], known);
  const g = num(snap.dA[0][i], gKnown);
  const terms = snap.aTerms[0][i].slice().sort((p, q) => p.j - q.j);
  const sample = XOR[snap.sample];
  return {
    title: `Input x${i + 1}`,
    subtitle: `Point (${sample.x[0]}, ${sample.x[1]}), target ${sample.y}`,
    intuition:
      `<p>An input is data, not something the network learns. ` +
      (known ? `For this point x${sub(i + 1)} = ${b(x)}.` : `Press Next to load the point.`) +
      `</p>` +
      (gKnown
        ? `<p>Backprop still computes a gradient for it, ${b(g)}, because the engine treats every value the same way. Nothing uses it: only weights and biases are updated.</p>`
        : `<p>When the backward pass reaches the first layer, the engine will compute a gradient here too, although nothing uses it.</p>`),
    math: [
      `\\fw{a^{(0)}_{${i + 1}}} = \\fw{x_{${i + 1}}} = ${x.x}`,
      `\\frac{\\partial \\tg{L}}{\\partial x_{${i + 1}}} = \\sum_j \\wt{w^{(1)}_{j${i + 1}}}\\,\\bk{\\delta^{(1)}_{j}} = ${terms.map((t) => `(${fmtTex(t.w)})${paren(num(t.delta, gKnown))}`).join('+')} = \\bk{${g.x}}`,
    ],
    code: `x = [Value(${sample.x[0]}), Value(${sample.x[1]})]   # leaf nodes: data\n# x[${i}].data = ${x.c}\n\n# after loss.backward():\n# x[${i}].grad = ${g.c}   (computed, never used)\nfor p in model.parameters():   # x is not in here\n    p.data -= lr * p.grad`,
    shared: [...(known ? [x.v] : []), ...(gKnown ? [g.v] : [])],
    param: null,
    neuron: null,
  };
}

export function explain(sel: Exclude<Selection, null>, ctx: ExplainContext): Explanation {
  if (sel.kind === 'weight') return explainWeight(ctx, sel.l, sel.j, sel.k);
  if (sel.kind === 'bias') return explainBias(ctx, sel.l, sel.j);
  if (sel.col === 0) return explainInput(ctx, sel.i);
  if (sel.col === 3) return explainOutput(ctx);
  return explainHidden(ctx, sel.col, sel.i);
}
