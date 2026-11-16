/**
 * The article under the stage. Plain language first, then the mechanics, always with the live numbers.
 * Text marked live (italic) and equations with a cyan border are recomputed from the store.
 */
import { XOR } from '../engine/data';
import { EPS, numericGrad, relativeError } from '../engine/gradcheck';
import { sampleLoss } from '../engine/matrix';
import type { ParamRef } from '../engine/params';
import { fmt, fmtSci, fmtTex } from '../model/format';
import type { Store } from '../model/store';
import { paramLabel } from '../scene/descent';
import { neuronFacts } from '../scene/exploded';
import { tex } from '../ui/katex';

const m = (src: string): string => tex(src, false);
const M = (src: string, note = ''): string => `<div class="eq">${tex(src)}${note ? `<small>${note}</small>` : ''}</div>`;
const live = (key: string): string => `<i class="live" data-k="${key}"></i>`;
const leq = (key: string, note = ''): string => `<div class="eq live-eq"><div data-eq="${key}"></div>${note ? `<small>${note}</small>` : ''}</div>`;
const show = (id: string, label: string): string => `<button type="button" class="show" data-show="${id}">${label} <span aria-hidden="true">↑</span></button>`;

const P = (x: number): string => `(${fmtTex(x)})`;
const vec = (xs: number[]): string => `\\begin{pmatrix}${xs.map((x) => fmtTex(x)).join('\\\\')}\\end{pmatrix}`;
const row = (xs: number[]): string => `\\begin{pmatrix}${xs.map((x) => fmtTex(x)).join('&')}\\end{pmatrix}`;
const mat = (rows: number[][]): string => `\\begin{pmatrix}${rows.map((r) => r.map((x) => fmtTex(x)).join('&')).join('\\\\')}\\end{pmatrix}`;

/** The weight used as the running example for gradients, updates and verification. */
const EXAMPLE: ParamRef = { kind: 'w', l: 2, j: 0, k: 0 };

function exampleRef(store: Store): ParamRef {
  const sel = store.s.selection;
  if (sel && sel.kind === 'weight') return { kind: 'w', l: sel.l, j: sel.j, k: sel.k };
  if (sel && sel.kind === 'bias') return { kind: 'b', l: sel.l, j: sel.j };
  return EXAMPLE;
}

const TEXT: Record<string, (s: Store) => string> = {
  point: (s) => `(${s.snap.x[0]}, ${s.snap.x[1]})`,
  target: (s) => String(s.snap.y),
  seed: (s) => String(s.s.seed),
  lr: (s) => fmt(s.s.lr, 2),
  yhat: (s) => fmt(s.snap.yhat),
  gap: (s) => fmt(s.snap.diff),
  loss: (s) => fmt(s.snap.loss),
  meanLoss: (s) => fmt(s.batch.loss),
  delta3: (s) => fmt(s.snap.delta[2][0]),
  a1: (s) => s.snap.a[1].map((v) => fmt(v)).join(', '),
  a2: (s) => s.snap.a[2].map((v) => fmt(v)).join(', '),
  nodes: (s) => String(s.snap.nodeCount),
  exName: (s) => (s.s.exploded.l === 2 ? 'the output neuron' : `hidden ${s.s.exploded.l + 1}, neuron ${s.s.exploded.j + 1}`),
  exConsumers: (s) => {
    const n = neuronFacts(s.snap, s.s.exploded.l, s.s.exploded.j).m;
    return n === 1 ? 'one consumer' : `${n} consumers`;
  },
  exGate: (s) => {
    const f = neuronFacts(s.snap, s.s.exploded.l, s.s.exploded.j);
    if (f.isOutput) return 'There is no ReLU on the output neuron, so the gradient passes through unchanged';
    return f.gate === 1 ? `Its z is ${fmt(f.z)}, positive, so the gate is open and the gradient passes through` : `Its z is ${fmt(f.z)}, not positive, so the gate is closed and the gradient stops here`;
  },
  wName: (s) => paramLabel(exampleRef(s)),
  seedSteps: (s) =>
    s.s.seedStatus.converges
      ? `reaches a mean loss below 0.001 in ${s.s.seedStatus.steps} steps at η = ${s.s.seedStatus.lr}`
      : `gets stuck near a mean loss of ${fmt(s.s.seedStatus.finalLoss, 2)} at η = ${s.s.seedStatus.lr}`,
  afterLoss: (s) => fmt(sampleLoss(s.after, s.s.sample)),
  numPlus: (s) => numericGrad(s.s.params, exampleRef(s), s.s.sample, EPS).lossPlus.toFixed(10),
  numMinus: (s) => numericGrad(s.s.params, exampleRef(s), s.s.sample, EPS).lossMinus.toFixed(10),
  numSlope: (s) => numericGrad(s.s.params, exampleRef(s), s.s.sample, EPS).numeric.toFixed(8),
  numBackprop: (s) => refGrad(s, exampleRef(s)).toFixed(8),
  numErr: (s) => fmtSci(relativeError(refGrad(s, exampleRef(s)), numericGrad(s.s.params, exampleRef(s), s.s.sample, EPS).numeric)),
};

function refGrad(s: Store, ref: ParamRef): number {
  const g = s.snap.grads;
  return ref.kind === 'w' ? g.W[ref.l][ref.j][ref.k] : g.b[ref.l][ref.j];
}

const EQ: Record<string, (s: Store) => string> = {
  // z and a of hidden 1, neuron 1
  fwdNeuron: (s) => {
    const p = s.snap.params;
    const x = s.snap.x;
    return `\\begin{aligned} z^{(1)}_1 &= \\wt{w^{(1)}_{11}}\\fw{x_1} + \\wt{w^{(1)}_{12}}\\fw{x_2} + \\wt{b^{(1)}_1} = ${P(p.W[0][0][0])}${P(x[0])} + ${P(p.W[0][0][1])}${P(x[1])} + ${P(p.b[0][0])} = ${fmtTex(s.snap.z[0][0])} \\\\ \\fw{a^{(1)}_1} &= \\max(0,\\ ${fmtTex(s.snap.z[0][0])}) = \\fw{${fmtTex(s.snap.a[1][0])}} \\end{aligned}`;
  },
  fwdMatrix: (s) => {
    const p = s.snap.params;
    return `z^{(1)} = ${mat(p.W[0])}${vec(s.snap.x)} + ${vec(p.b[0])} = ${vec(s.snap.z[0])}, \\qquad \\fw{a^{(1)}} = \\fw{${vec(s.snap.a[1])}}`;
  },
  loss: (s) => `\\tg{L} = (\\fw{\\hat y} - \\tg{y})^2 = (${fmtTex(s.snap.yhat)} - ${s.snap.y})^2 = (${fmtTex(s.snap.diff)})^2 = \\tg{${fmtTex(s.snap.loss)}}`,
  meanLoss: (s) =>
    `\\tg{\\bar L} = \\tfrac14\\big(${s.batch.samples.map((x) => fmtTex(x.loss)).join(' + ')}\\big) = \\tg{${fmtTex(s.batch.loss)}}`,
  delta3: (s) => `\\bk{\\delta^{(3)}} = 2(\\fw{\\hat y} - \\tg{y}) = 2${P(s.snap.diff)} = \\bk{${fmtTex(s.snap.delta[2][0])}}`,
  // delta of hidden 2, neuron 1 and hidden 1, neuron 1
  delta2: (s) => {
    const t = s.snap.aTerms[2][0][0];
    return `\\bk{\\delta^{(2)}_1} = \\wt{w^{(3)}_{11}}\\,\\bk{\\delta^{(3)}}\\,\\varphi'(z^{(2)}_1) = ${P(t.w)}${P(t.delta)}(${s.snap.dAct[1][0]}) = \\bk{${fmtTex(s.snap.delta[1][0])}}`;
  },
  delta1: (s) => {
    const terms = s.snap.aTerms[1][0].slice().sort((a, b) => a.j - b.j);
    return `\\begin{aligned} \\bk{\\delta^{(1)}_1} &= \\big(${terms.map((t) => `${P(t.w)}${P(t.delta)}`).join(' + ')}\\big)(${s.snap.dAct[0][0]}) \\\\ &= ${P(s.snap.dA[1][0])}(${s.snap.dAct[0][0]}) = \\bk{${fmtTex(s.snap.delta[0][0])}} \\end{aligned}`;
  },
  deltaMatrix: (s) => {
    const p = s.snap.params;
    const wT = p.W[2][0].map((w) => [w]);
    return `\\bk{\\delta^{(2)}} = \\big(W^{(3)\\top}\\bk{\\delta^{(3)}}\\big) \\odot \\varphi'(z^{(2)}) = \\Big(${mat(wT)}${P(s.snap.delta[2][0])}\\Big) \\odot ${vec(s.snap.dAct[1])} = \\bk{${vec(s.snap.delta[1])}}`;
  },
  gradMatrix: (s) =>
    `\\frac{\\partial \\tg{L}}{\\partial W^{(3)}} = \\bk{\\delta^{(3)}}\\,\\fw{a^{(2)\\top}} = ${P(s.snap.delta[2][0])}\\,${row(s.snap.a[2])} = \\bk{${row(s.snap.grads.W[2][0])}}`,
  // the opened neuron
  exAccum: (s) => {
    const f = neuronFacts(s.snap, s.s.exploded.l, s.s.exploded.j);
    const terms = f.terms;
    if (f.isOutput) return `\\frac{\\partial \\tg{L}}{\\partial \\hat y} = 2(\\hat y - y) = \\bk{${fmtTex(f.dA)}}`;
    return `\\begin{aligned} \\frac{\\partial \\tg{L}}{\\partial a} &= ${terms.map((t) => `\\underbrace{${P(t.w)}${P(t.delta)}}_{\\text{to neuron ${t.j + 1}}}`).join(' + ')} \\\\ &= ${terms.map((t) => P(t.amount)).join(' + ')} = \\bk{${fmtTex(f.dA)}} \\end{aligned}`;
  },
  exRules: (s) => {
    const f = neuronFacts(s.snap, s.s.exploded.l, s.s.exploded.j);
    return `\\begin{aligned} \\bk{\\delta} &= \\varphi'(z)\\cdot\\frac{\\partial L}{\\partial a} = (${f.gate})${P(f.dA)} = \\bk{${fmtTex(f.delta)}} \\\\ \\frac{\\partial L}{\\partial b} &= \\bk{\\delta}\\cdot 1 = \\bk{${fmtTex(f.dB)}} \\\\ \\frac{\\partial L}{\\partial w_1} &= \\bk{\\delta}\\cdot \\fw{a_1} = ${P(f.delta)}${P(f.aIn[0])} = \\bk{${fmtTex(f.dW[0])}} \\end{aligned}`;
  },
  update: (s) => {
    const ref = exampleRef(s);
    const p = s.snap.params;
    const w = ref.kind === 'w' ? p.W[ref.l][ref.j][ref.k] : p.b[ref.l][ref.j];
    const g = refGrad(s, ref);
    return `w \\leftarrow w - \\eta\\,\\frac{\\partial \\tg{L}}{\\partial w} = ${fmtTex(w)} - ${P(s.s.lr)}${P(g)} = \\wt{${fmtTex(w - s.s.lr * g)}}`;
  },
  batch: (s) => {
    const ref = exampleRef(s);
    const pick = (g: { W: number[][][]; b: number[][] }): number => (ref.kind === 'w' ? g.W[ref.l][ref.j][ref.k] : g.b[ref.l][ref.j]);
    const gs = s.batch.samples.map((x) => pick(x.grads));
    return `\\frac{\\partial \\tg{\\bar L}}{\\partial w} = \\frac14\\big(${gs.map((g) => P(g)).join(' + ')}\\big) = \\bk{${fmtTex(pick(s.batch.grads))}}`;
  },
};

function xorTable(s: Store): string {
  const rows = XOR.map((p, i) => {
    const t = s.batch.samples[i];
    return `<tr><td>${i + 1}${i === s.s.sample ? ' · on stage' : ''}</td><td>${p.x[0]}</td><td>${p.x[1]}</td><td>${p.y}</td><td><i class="live">${fmt(t.yhat)}</i></td><td><i class="live">${fmt(t.loss)}</i></td></tr>`;
  }).join('');
  return `<table><thead><tr><th>Point</th><th>x₁</th><th>x₂</th><th>target y</th><th>network says ŷ</th><th>loss</th></tr></thead><tbody>${rows}</tbody></table>`;
}

const HTML = `
<h2>How twenty-five numbers learn XOR</h2>
<p class="lead">This network is nothing more than <b>25 numbers</b>: 18 weights and 7 biases. Training means nudging those numbers until four inputs give four right answers. <b>Backpropagation</b> is the bookkeeping that tells every one of the 25 which way to move and by how much.</p>
<p>Everything on this page uses the live numbers from the stage above. Right now that is point x = ${live('point')} with target ${live('target')}, starting weights from seed ${live('seed')}, and learning rate η = ${live('lr')}. Change something up there and the numbers down here change with it. Numbers in <i class="live">italics</i> and equations with a cyan border are live.</p>

<h3>What you see</h3>
<p>A small machine on a bench. The <b>spheres</b> are neurons, in four columns: two inputs, two hidden layers of three, and one output. The <b>tubes</b> are weights: a thick tube is a large weight, <span class="c-pos">green</span> is positive and <span class="c-neg">rose</span> is negative. The small ball riding on each neuron is its <b>bias</b>.</p>
<p>The task is <b>XOR</b>: answer 1 when exactly one of the two inputs is 1, and 0 otherwise. It is the classic small problem that a single neuron cannot solve, because no single straight line separates the two 1s from the two 0s. Here is how the network is doing on all four points at this moment:</p>
<div data-html="xor"></div>

<h3>What you can do</h3>
<p>Press <b>Next</b> to walk one training iteration in 12 stages: forward, loss, backward, update. <b>Click</b> any sphere, tube or small ball for a card with its numbers, in three layers: Intuition, Math, Code. Switch to <b>Train</b> to run the whole loop and watch the loss curve fall. Switch the view to <b>Exploded</b> to open one neuron into its individual operations. In the <b>Descent</b> tile, pick two weights and see the loss as a landscape. Press <b>Verify</b> on a weight to check backprop against a direct measurement.</p>

<h2>What a network is: the forward pass</h2>
<p>Picture each neuron as a small committee taking a vote. Every voice coming in has a <b>weight</b>: how much the committee trusts it, and whether it counts for or against. The committee also has a <b>bias</b>, its own leaning before anyone speaks. It adds everything up and announces one number.</p>
<p>Written down, neuron ${m('j')} in layer ${m('l')} computes a weighted sum:</p>
${M(`z^{(l)}_j = \\sum_k \\wt{w^{(l)}_{jk}}\\,\\fw{a^{(l-1)}_k} + \\wt{b^{(l)}_j}`)}
<p>Here ${m('\\fw{a^{(l-1)}_k}')} is the output of neuron ${m('k')} in the layer before (for the first layer these are the inputs, ${m('\\fw{a^{(0)}} = x')}), ${m('\\wt{w^{(l)}_{jk}}')} is the weight on the tube from neuron ${m('k')} to neuron ${m('j')}, ${m('\\wt{b^{(l)}_j}')} is the bias, and ${m('z^{(l)}_j')} is the sum.</p>
<p>Then comes the one non-straight step. The hidden neurons pass their sum through <b>ReLU</b>, written ${m('\\varphi')}: keep it if it is positive, otherwise output 0.</p>
${M(`\\fw{a^{(l)}_j} = \\varphi(z^{(l)}_j) = \\max(0,\\ z^{(l)}_j)`)}
<p>Without that kink, three layers of weighted sums would collapse into a single weighted sum, and one weighted sum cannot do XOR. For the top neuron of hidden 1, with the point on stage:</p>
${leq('fwdNeuron')}
<p>Each layer feeds the next. Hidden 1 outputs ${live('a1')}; hidden 2 takes those as its inputs and outputs ${live('a2')}. The <b>output neuron</b> is the same weighted sum but with no ReLU, so it can say any number: ŷ = ${live('yhat')}.</p>
${show('forward', 'Show me the forward pass')}
<h3>The same thing, a layer at a time</h3>
<p>Stack a layer's sums into a column and the weights into a table, and one layer is one line. ${m('W^{(1)}')} has 3 rows and 2 columns (3 neurons, 2 inputs), ${m('W^{(2)}')} is 3 by 3, ${m('W^{(3)}')} is 1 by 3.</p>
${M(`z^{(l)} = W^{(l)}\\,\\fw{a^{(l-1)}} + b^{(l)}, \\qquad \\fw{a^{(l)}} = \\varphi(z^{(l)}), \\qquad \\fw{\\hat y} = z^{(3)}`)}
${leq('fwdMatrix', 'Layer 1 for the point on stage. Row 1 is the neuron worked out above.')}

<h2>Loss: how wrong is it</h2>
<p>A dart lands somewhere on the board. Before you can improve your throw you need one number that says how bad it was. For a network that number is the <b>loss</b>. Here it is the gap between the answer ${m('\\fw{\\hat y}')} and the target ${m('\\tg{y}')}, squared:</p>
${leq('loss')}
<p>Squaring does two jobs. It makes the loss positive whether the answer was too high or too low, and it punishes a big miss much more than a small one. On the stage the gap is the bar on the rail next to the output, and the loss is the <b>area of the square</b> built on that bar.</p>
<p>That is the loss for one point. Training cares about all four, so it uses their average, the <b>mean loss</b>:</p>
${leq('meanLoss')}
${show('loss', 'Show me the loss')}

<h2>Gradient descent: why the negative gradient</h2>
<p>You are on a hillside in thick fog and want to reach the valley. You cannot see it, but you can feel the slope under your feet. So you take a step downhill, feel again, and repeat.</p>
<p>The loss is that hillside, except that instead of two directions to walk in there are 25: one for each weight and bias. For each of them, the <b>partial derivative</b> ${m('\\partial \\tg{L} / \\partial w')} answers one question: if I raise this number a little and leave the rest alone, how fast does the loss rise? The list of all 25 answers is the <b>gradient</b>. It points uphill, so the update steps the opposite way:</p>
${M(`w \\leftarrow w - \\eta\\,\\frac{\\partial \\tg{L}}{\\partial w}`)}
<p>${m('\\eta')} is the <b>learning rate</b>, the length of the step. Too small and the walk takes forever. Too large and you step clean across the valley and land higher up the other side. Open the Descent tile, raise η towards 1 and watch the cyan arrow land above where it started.</p>
<p class="small">The landscape in that tile is a slice: two parameters vary and the other 23 are held still. The real surface has 25 directions and cannot be drawn.</p>
${show('descent', 'Show me the landscape')}

<h2>Backprop on one neuron: the chain rule</h2>
<p>The answer came out wrong. Who is to blame? Backpropagation hands out blame fairly, and it works because a neuron is built from three tiny operations, each with a one-line rule for passing blame backward. "Blame" has a precise meaning here: the <b>gradient</b> of a value is how much the loss changes per unit change of that value.</p>
<p>The rule that glues the steps together is the <b>chain rule</b>. If the loss depends on ${m('z')} only through ${m('a')}, then</p>
${M(`\\frac{\\partial \\tg{L}}{\\partial z} = \\underbrace{\\frac{\\partial a}{\\partial z}}_{\\text{local slope}} \\cdot \\underbrace{\\frac{\\partial \\tg{L}}{\\partial a}}_{\\text{arriving from downstream}}`)}
<p>Every operation only needs to know its own local slope. The three that make up a neuron:</p>
<ul>
<li><b>Add passes it along.</b> For ${m('s = u + v')}, both local slopes are 1, so ${m('u')} and ${m('v')} each receive the incoming gradient unchanged.</li>
<li><b>Multiply swaps.</b> For ${m('p = w \\cdot a')}, the slope with respect to ${m('w')} is ${m('a')}, and with respect to ${m('a')} it is ${m('w')}. Each side gets the incoming gradient times the <i>other</i> side.</li>
<li><b>ReLU is a gate.</b> Its slope is 1 where ${m('z > 0')} and 0 elsewhere. The gradient either passes untouched or stops.</li>
</ul>
<p>Open ${live('exName')} in the Exploded view and step through it. Its output has ${live('exConsumers')}, and each one sends back its own gradient times the connecting weight. A value that feeds several places affects the loss along several paths, and the paths <b>add</b>:</p>
${leq('exAccum')}
<p>In the engine this is one small word, <code>+=</code>. Here is the multiply rule from microtensor, the Python engine this app mirrors:</p>
<pre>def _backward():
    self.grad  += other.data * out.grad   <span class="cm"># each side gets</span>
    other.grad += self.data  * out.grad   <span class="cm"># the other side</span></pre>
<p>Writing <code>=</code> instead of <code>+=</code> there would keep only the last path and silently give wrong gradients. With the output's gradient complete, the rest of the neuron follows from the three rules. ${live('exGate')}. We give the gradient at ${m('z')} its own name, ${m('\\bk{\\delta} = \\partial \\tg{L}/\\partial z')}, because everything else is built from it:</p>
${leq('exRules')}
${show('exploded', 'Open this neuron')}

<h2>Backprop through layers: the δ recursion</h2>
<p>One neuron is settled. The network is the same move repeated, starting from the only place where blame is known directly: the output. The slope of a square is twice its side, so</p>
${leq('delta3')}
<p>Now step one layer back. A hidden neuron ${m('k')} in layer ${m('l')} collects ${m('w \\cdot \\delta')} from every neuron ${m('j')} it feeds in layer ${m('l+1')}, adds them, and passes the total through its ReLU gate:</p>
${M(`\\bk{\\delta^{(l)}_k} = \\Big(\\sum_j \\wt{w^{(l+1)}_{jk}}\\,\\bk{\\delta^{(l+1)}_j}\\Big)\\,\\varphi'(z^{(l)}_k)`)}
<p>For the top neuron of hidden 2, which feeds only the output, and then the top neuron of hidden 1, which feeds all three neurons of hidden 2:</p>
${leq('delta2')}
${leq('delta1')}
<p>This is why the pass runs <b>backward</b>. Each δ needs the δ of every neuron it feeds, so layer 3 must be finished before layer 2, and layer 2 before layer 1. Each value is computed once and reused by everything behind it. Once a neuron's δ is known, the gradients of its own weights and bias cost one multiplication each:</p>
${M(`\\frac{\\partial \\tg{L}}{\\partial \\wt{w^{(l)}_{jk}}} = \\bk{\\delta^{(l)}_j}\\,\\fw{a^{(l-1)}_k}, \\qquad \\frac{\\partial \\tg{L}}{\\partial \\wt{b^{(l)}_j}} = \\bk{\\delta^{(l)}_j}`)}
<p>A weight is blamed in proportion to the signal it carried. If the activation coming in was 0, the weight's gradient is exactly 0: it passed nothing on, so it changed nothing.</p>
${show('delta1', 'Show me the backward pass')}
<h3>The same thing, a layer at a time</h3>
<p>Stack each layer's δ into a column and the recursion becomes one line per layer. ${m('\\odot')} means multiply element by element. The transpose ${m('W^{\\top}')} appears because going backward we travel the same tubes in the other direction, so rows and columns swap roles.</p>
${M(`\\bk{\\delta^{(3)}} = 2(\\fw{\\hat y} - \\tg{y}), \\qquad \\bk{\\delta^{(l)}} = \\big(W^{(l+1)\\top}\\,\\bk{\\delta^{(l+1)}}\\big) \\odot \\varphi'(z^{(l)})`)}
${M(`\\frac{\\partial \\tg{L}}{\\partial W^{(l)}} = \\bk{\\delta^{(l)}}\\,\\fw{a^{(l-1)\\top}}, \\qquad \\frac{\\partial \\tg{L}}{\\partial b^{(l)}} = \\bk{\\delta^{(l)}}`)}
${leq('deltaMatrix', "Layer 2's δ from layer 3's. The column of 1s and 0s is the ReLU gates.")}
${leq('gradMatrix', 'The three gradients of the last layer in one product.')}
<h3>Doing it by hand</h3>
<ol>
<li>Forward: compute every ${m('z')} and ${m('a')}, layer by layer, and keep them all.</li>
<li>Loss: ${m('L = (\\hat y - y)^2')}.</li>
<li>Output: ${m('\\delta^{(3)} = 2(\\hat y - y)')}.</li>
<li>For layer 2, then layer 1: for each neuron, add up ${m('w \\cdot \\delta')} over the neurons it feeds, then multiply by 1 if its ${m('z')} was positive and by 0 if not.</li>
<li>For every weight: ${m('\\delta')} of the neuron it feeds times the activation it carried. For every bias: ${m('\\delta')} itself.</li>
<li>Update: subtract ${m('\\eta')} times the gradient from each of the 25 numbers.</li>
</ol>
<p>Pick a point, write the numbers down from the forward stages, and check each of your results against the cards.</p>

<h2>The update and the training loop</h2>
<p>All 25 gradients are known, so all 25 numbers move at once. For ${live('wName')} (click another tube or bias to change the example):</p>
${leq('update')}
<p>After this one update the loss on this point would go from ${live('loss')} to ${live('afterLoss')}. In Step mode each iteration looks at one point only. That fixes this point a little and may make the other three slightly worse.</p>
<p>Train mode is fairer. It runs all four points through the network, and because the mean loss is an average, its gradient is simply the <b>mean of the four per-point gradients</b>. For the same parameter:</p>
${leq('batch')}
<p>Then it is a loop: forward, loss, backward, update, again. From these starting weights it ${live('seedSteps')}. The loss curve shows the mean loss falling, and the decision map shows the network's answer across the whole unit square, slowly turning bright near the two points that want 1 and dark near the two that want 0.</p>
<p>It does not always work. With plain gradient descent and only three hidden neurons per layer, about <b>80 out of 100 random starts</b> reach the target; the rest get stuck with neurons whose ReLU is closed for every point that still needs fixing, so no gradient reaches them. (Measured over 400 random starts at η = 0.1, 3000 steps each.) The four named presets are known to converge. <b>Randomise</b> draws a fresh start, checks it, and tells you when it is one of the stuck ones.</p>
${show('train', 'Show me training')}

<h2>Verifying gradients</h2>
<p>If you do not trust the bookkeeping, measure the slope directly. Nudge one number up by a tiny ${m('\\varepsilon')}, then down by ${m('\\varepsilon')}, run the forward pass both times and see how far the loss moved:</p>
${M(`\\frac{\\partial \\tg{L}}{\\partial w} \\approx \\frac{\\tg{L}(w + \\varepsilon) - \\tg{L}(w - \\varepsilon)}{2\\,\\varepsilon}`)}
<p>For ${live('wName')} on the point on stage, with ε = 0.00001: the loss is ${live('numPlus')} after the nudge up and ${live('numMinus')} after the nudge down. That gives a slope of ${live('numSlope')}. Backprop says ${live('numBackprop')}. The relative error, ${m('|a - n| / (|a| + |n|)')}, is ${live('numErr')}.</p>
<p>The two agree to many digits, and that is the point. Backprop is not an estimate. It is the <b>exact derivative</b>, and it is cheap: the nudge needs two forward passes for <i>each</i> number, 50 passes for all 25, while backprop gets all 25 from one forward and one backward pass. On a network with a billion weights that difference is the whole game.</p>
<p class="small">One honest caveat: ReLU has a corner at z = 0 where the slope is not defined. If a nudge pushes some z across that corner, the measured slope and backprop can disagree. The Verify card warns you when a z is that close to zero.</p>
${show('verify', 'Show me the check')}

<h2>How accurate it is</h2>
<p>Every number you see, on the stage, in the cards and on this page, is computed by a small scalar autograd engine written for this app. It mirrors microtensor line for line: a <code>Value</code> that remembers which operation made it, a topological sort, and <code>+=</code> in every backward rule. For the point on stage the engine's graph has ${live('nodes')} nodes.</p>
<p>The engine is tested. Its gradients match the nudge method on all 25 parameters with a relative error under one in a million, and they match a separate, hand-written matrix version of the equations above to 12 decimal places. The three layers of each card are built from the same numbers, and a test checks that they agree.</p>
<p>What is simplified or bent for the sake of the picture:</p>
<ul>
<li><b>Rounding.</b> The stage shows 2 decimals and the cards 3. Multiply the rounded numbers yourself and the last digit can differ from what is shown.</li>
<li><b>Tube thickness</b> follows the square root of |w| and stops growing at |w| = 2. The loss rail only spans −1.2 to 2.2; an answer outside that is drawn at the end of the rail.</li>
<li><b>Amber brightness</b> is relative to the largest gradient in the current pass (or to 0.05, if every gradient is smaller than that). It compares tubes with each other, not one iteration with the next.</li>
<li><b>Pulses</b> are animation. How long they take means nothing.</li>
<li><b>The Exploded view</b> draws one sum node. The engine really adds with a chain of two-input additions; each one passes the gradient along unchanged, so the picture is equivalent.</li>
<li><b>The Descent landscape</b> is a 2-D slice of a 25-D surface, with its height compressed (a log scale) so small losses stay visible. The cyan arrow is the step along those two parameters only; the real update moves all 25 at once, which also reshapes the slice.</li>
<li><b>The decision map and the landscape</b> are grids of plain-number forward passes, not engine graphs. A test pins that fast path to the engine exactly.</li>
<li><b>ReLU at exactly z = 0</b> is given slope 0, as in microtensor.</li>
<li><b>Biases start between 0 and 0.5</b>, not at 0 as in microtensor. With zero biases the point (0, 0) puts every first-layer z exactly on the ReLU corner, and far more random starts get stuck.</li>
<li><b>The Verify wobble</b> is exaggerated. The real nudge is 0.00001.</li>
<li><b>Step mode learns from one point at a time; Train mode uses all four.</b> There is no momentum, no Adam, no mini-batches and no regularisation. Real networks add those, are millions of times larger and often use other losses, but the backward pass is this one.</li>
</ul>

<h2>Keys</h2>
<p class="keys"><kbd>→</kbd> or <kbd>N</kbd> next stage · <kbd>←</kbd> or <kbd>B</kbd> back · <kbd>Space</kbd> auto-play, or run and pause in Train mode · <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> <kbd>4</kbd> choose the XOR point · <kbd>T</kbd> Step or Train · <kbd>⇧V</kbd> Lab, Exploded, Studio · <kbd>X</kbd> open the selected neuron · <kbd>V</kbd> verify the selected weight · <kbd>[</kbd> <kbd>]</kbd> learning rate · <kbd>G</kbd> randomise the weights · <kbd>?</kbd> legend · <kbd>Tab</kbd> walk the objects in the scene, <kbd>Enter</kbd> to open one, <kbd>Esc</kbd> to close · <kbd>C</kbd> cinematic cameras · <kbd>R</kbd> reset the view · drag to orbit · right-drag to pan · scroll to zoom · <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> to move, <kbd>Q</kbd> <kbd>E</kbd> to turn · <kbd>PgDn</kbd> to come down here.</p>
<p class="small">The keys act on the stage, so they are switched off while you are scrolled down in the article.</p>
`;

export class Article {
  private texts: { el: HTMLElement; fn: (s: Store) => string; last: string }[] = [];
  private eqs: { el: HTMLElement; fn: (s: Store) => string; last: string }[] = [];
  private htmls: { el: HTMLElement; last: string }[] = [];
  private dirty = true;
  private timer = 0;

  constructor(
    private root: HTMLElement,
    private store: Store,
    onShow: (id: string) => void,
  ) {
    root.innerHTML = HTML;
    root.querySelectorAll<HTMLElement>('[data-k]').forEach((el) => {
      const fn = TEXT[el.dataset.k!];
      if (!fn) throw new Error(`article: unknown live key ${el.dataset.k}`);
      this.texts.push({ el, fn, last: '' });
    });
    root.querySelectorAll<HTMLElement>('[data-eq]').forEach((el) => {
      const fn = EQ[el.dataset.eq!];
      if (!fn) throw new Error(`article: unknown live equation ${el.dataset.eq}`);
      this.eqs.push({ el, fn, last: '' });
    });
    root.querySelectorAll<HTMLElement>('[data-html]').forEach((el) => this.htmls.push({ el, last: '' }));
    root.addEventListener('click', (e) => {
      const b = (e.target as HTMLElement).closest<HTMLElement>('[data-show]');
      if (b) onShow(b.dataset.show!);
    });
    store.subscribe(() => {
      this.dirty = true;
    });
    this.update();
  }

  /** Called every frame; refreshes at most a few times a second, and only while the article is on screen. */
  frame(dt: number): void {
    this.timer += dt;
    if (!this.dirty || this.timer < 0.2) return;
    if (this.root.getBoundingClientRect().top > window.innerHeight) return;
    this.update();
  }

  update(): void {
    this.dirty = false;
    this.timer = 0;
    const s = this.store;
    for (const t of this.texts) {
      const v = t.fn(s);
      if (v !== t.last) {
        t.el.textContent = v;
        t.last = v;
      }
    }
    for (const e of this.eqs) {
      const src = e.fn(s);
      if (src !== e.last) {
        e.el.innerHTML = tex(src);
        e.last = src;
      }
    }
    for (const h of this.htmls) {
      const v = xorTable(s);
      if (v !== h.last) {
        h.el.innerHTML = v;
        h.last = v;
      }
    }
  }
}
