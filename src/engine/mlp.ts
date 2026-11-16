/**
 * The network as a graph of Values. Mirrors microtensor/nn.py:
 *
 *   val = self.b
 *   for i in range(len(x)):
 *       val += self.w[i] * x[i]
 *   return val.relu() if self.nonlin else val
 *
 * The only difference is that the intermediate nodes are kept so views can read them.
 */
import { N_LAYERS, SIZES, type Activation, type Params } from './params';
import { Value } from './value';

export type NeuronNodes = {
  w: Value[];
  b: Value;
  /** The inputs this neuron read (activations of the previous layer). */
  x: Value[];
  /** prod[k] = w[k] * x[k] */
  prod: Value[];
  /** sums[k] = b + prod[0] + ... + prod[k]; the last one is z. */
  sums: Value[];
  z: Value;
  /** Output. For the linear output neuron this is the same node as z. */
  a: Value;
  nonlin: boolean;
};

export type ParamNodes = { W: Value[][][]; b: Value[][] };

export function paramNodes(p: Params): ParamNodes {
  return {
    W: p.W.map((layer, l) => layer.map((row, j) => row.map((w, k) => new Value(w).named(`w${l + 1}[${j}][${k}]`)))),
    b: p.b.map((layer, l) => layer.map((b, j) => new Value(b).named(`b${l + 1}[${j}]`))),
  };
}

function neuronForward(
  w: Value[],
  b: Value,
  x: Value[],
  nonlin: boolean,
  act: Activation,
  tag: string,
): NeuronNodes {
  const prod: Value[] = [];
  const sums: Value[] = [];
  let val = b;
  for (let i = 0; i < x.length; i++) {
    const p = w[i].mul(x[i]).named(`${tag}.prod${i}`);
    prod.push(p);
    val = val.add(p).named(`${tag}.sum${i}`);
    sums.push(val);
  }
  const z = val.named(`z${tag}`);
  const a = nonlin ? (act === 'relu' ? z.relu() : z.tanh()).named(`a${tag}`) : z;
  return { w, b, x, prod, sums, z, a, nonlin };
}

export type ForwardNodes = {
  inputs: Value[];
  /** layers[l][j], l = 0..2 */
  layers: NeuronNodes[][];
  yhat: Value;
};

export function forwardNodes(pn: ParamNodes, x: readonly number[], act: Activation = 'relu'): ForwardNodes {
  const inputs = x.map((v, i) => new Value(v).named(`x[${i}]`));
  let cur = inputs;
  const layers: NeuronNodes[][] = [];
  for (let l = 0; l < N_LAYERS; l++) {
    const isLast = l === N_LAYERS - 1;
    const layer: NeuronNodes[] = [];
    for (let j = 0; j < SIZES[l + 1]; j++) {
      layer.push(neuronForward(pn.W[l][j], pn.b[l][j], cur, !isLast, act, `${l + 1}[${j}]`));
    }
    layers.push(layer);
    cur = layer.map((n) => n.a);
  }
  return { inputs, layers, yhat: cur[0] };
}

export type LossNodes = { target: Value; diff: Value; loss: Value };

/** L = (yhat - y) ** 2, written with the same derived ops as microtensor (sub = add(neg)). */
export function lossNodes(yhat: Value, y: number): LossNodes {
  const target = new Value(y).named('y');
  const diff = yhat.sub(target).named('diff');
  const loss = diff.pow(2).named('L');
  return { target, diff, loss };
}
