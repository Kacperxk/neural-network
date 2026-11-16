/**
 * Run the engine and copy everything it computed into plain, immutable data.
 * The UI reads numbers only from these snapshots: nothing is re-derived with a second formula.
 */
import { XOR } from './data';
import { forwardNodes, lossNodes, paramNodes } from './mlp';
import { N_LAYERS, SIZES, zerosLike, type Activation, type Params } from './params';
import { Value } from './value';

/** One `+=` received by an activation: sent back by neuron j of the next layer. */
export type ATerm = { j: number; w: number; delta: number; amount: number };

export type Snapshot = {
  /** The parameters this trace was run at (the values held by the graph's leaf nodes). */
  params: Params;
  sample: number;
  x: number[];
  y: number;
  /** a[0] = x, a[l + 1][j] = output of neuron j in weight-layer l. */
  a: number[][];
  /** z[l][j], pre-activation. */
  z: number[][];
  /** prod[l][j][k] = W[l][j][k] * a[l][k] */
  prod: number[][][];
  /** Local derivative of the activation at z[l][j]. 1 for the linear output neuron. */
  dAct: number[][];
  yhat: number;
  diff: number;
  loss: number;
  /** dL/dyhat's local factor: d(diff**2)/d(diff) = 2 * diff. */
  dLossdDiff: number;
  /** delta[l][j] = dL/dz[l][j] (z.grad in the engine). */
  delta: number[][];
  /** dA[l][k] = dL/da[l][k] (a.grad); dA[0] is the gradient on the inputs. */
  dA: number[][];
  /** aTerms[l][k]: the `+=` contributions that add up to dA[l][k], in the order the engine applied them. */
  aTerms: ATerm[][][];
  /** dProd[l][j][k]: gradient on each product node (what the sum hands to each of its inputs). */
  dProd: number[][][];
  grads: Params;
  /** Number of nodes in the graph and in the backward order, for the "how accurate" section. */
  nodeCount: number;
};

export function traceSample(p: Params, sampleIndex: number, act: Activation = 'relu'): Snapshot {
  const s = XOR[sampleIndex];
  const pn = paramNodes(p);
  const fwd = forwardNodes(pn, s.x, act);
  const ln = lossNodes(fwd.yhat, s.y);
  const order = ln.loss.backward();

  const a: number[][] = [fwd.inputs.map((v) => v.data)];
  const z: number[][] = [];
  const prod: number[][][] = [];
  const dAct: number[][] = [];
  const delta: number[][] = [];
  const dProd: number[][][] = [];
  const dA: number[][] = [fwd.inputs.map((v) => v.grad)];
  const grads = zerosLike();

  for (let l = 0; l < N_LAYERS; l++) {
    const layer = fwd.layers[l];
    z.push(layer.map((n) => n.z.data));
    a.push(layer.map((n) => n.a.data));
    prod.push(layer.map((n) => n.prod.map((v) => v.data)));
    dAct.push(layer.map((n) => (n.nonlin ? n.a.local[0] : 1)));
    delta.push(layer.map((n) => n.z.grad));
    dProd.push(layer.map((n) => n.prod.map((v) => v.grad)));
    dA.push(layer.map((n) => n.a.grad));
    for (let j = 0; j < SIZES[l + 1]; j++) {
      grads.b[l][j] = pn.b[l][j].grad;
      for (let k = 0; k < SIZES[l]; k++) grads.W[l][j][k] = pn.W[l][j][k].grad;
    }
  }

  // Which consumer sent each `+=` to an activation: look the sender up among the next layer's products.
  const aTerms: ATerm[][][] = [];
  for (let l = 0; l < N_LAYERS; l++) {
    const sources: Value[] = l === 0 ? fwd.inputs : fwd.layers[l - 1].map((n) => n.a);
    aTerms.push(
      sources.map((src, k) =>
        src.terms.map((t) => {
          const j = fwd.layers[l].findIndex((n) => n.prod[k] === t.from);
          if (j < 0) throw new Error(`unexpected consumer of ${src.label}: ${t.from.label}`);
          return { j, w: pn.W[l][j][k].data, delta: fwd.layers[l][j].z.grad, amount: t.amount };
        }),
      ),
    );
  }

  return {
    params: {
      W: pn.W.map((layer) => layer.map((row) => row.map((v) => v.data))),
      b: pn.b.map((layer) => layer.map((v) => v.data)),
    },
    sample: sampleIndex,
    x: s.x.slice(),
    y: s.y,
    a,
    z,
    prod,
    dAct,
    yhat: fwd.yhat.data,
    diff: ln.diff.data,
    loss: ln.loss.data,
    dLossdDiff: ln.loss.local[0],
    delta,
    dA,
    aTerms,
    dProd,
    grads,
    nodeCount: order.length,
  };
}

export type BatchTrace = {
  /** Mean loss over the 4 samples. */
  loss: number;
  /** Gradient of the mean loss, from ONE graph that contains all four samples. */
  grads: Params;
  /** The four per-sample traces, each from its own graph. */
  samples: Snapshot[];
};

/**
 * Full-batch loss L = (L0 + L1 + L2 + L3) * 0.25 as a single graph sharing the parameter nodes.
 * The parameters feed four copies of the network, so their gradients accumulate with `+=` across samples.
 */
export function traceBatch(p: Params, act: Activation = 'relu'): BatchTrace {
  const pn = paramNodes(p);
  let total: Value | null = null;
  for (const s of XOR) {
    const fwd = forwardNodes(pn, s.x, act);
    const { loss } = lossNodes(fwd.yhat, s.y);
    total = total ? total.add(loss) : loss;
  }
  const mean = total!.mul(1 / XOR.length);
  mean.backward();

  const grads = zerosLike();
  for (let l = 0; l < N_LAYERS; l++) {
    for (let j = 0; j < SIZES[l + 1]; j++) {
      grads.b[l][j] = pn.b[l][j].grad;
      for (let k = 0; k < SIZES[l]; k++) grads.W[l][j][k] = pn.W[l][j][k].grad;
    }
  }
  return { loss: mean.data, grads, samples: XOR.map((_, i) => traceSample(p, i, act)) };
}
