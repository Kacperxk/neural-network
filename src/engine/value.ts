/**
 * Scalar autograd. A TypeScript mirror of microtensor/engine.py (see /reference).
 *
 * Same structure as the Python original:
 *  - every op creates an output Value that remembers its children and its op,
 *  - every op installs a `_backward` closure that pushes gradient to the children with `+=`,
 *  - `backward()` topologically sorts the graph and runs the closures in reverse.
 *
 * Two additions, both only so the UI can *show* what the engine did instead of recomputing it:
 *  - `local[i]` keeps the local derivative d(out)/d(child i) that the closure multiplies by,
 *  - `terms` records each `+=` contribution a value received, and from which consumer.
 */

export type Term = {
  /** The consumer node that sent this contribution. */
  from: Value;
  /** local derivative * consumer.grad, i.e. the amount added by one `+=`. */
  amount: number;
};

export class Value {
  data: number;
  grad = 0;
  readonly children: readonly Value[];
  readonly op: string;
  /** d(this)/d(children[i]), filled in when the op runs. */
  local: number[] = [];
  /** Every `+=` this value received during the last backward(), in arrival order. */
  terms: Term[] = [];
  label = '';
  _backward: () => void = () => {};

  constructor(data: number, children: readonly Value[] = [], op = '') {
    this.data = data;
    this.children = children;
    this.op = op;
  }

  named(label: string): this {
    this.label = label;
    return this;
  }

  /** `self.grad += amount` with a note of who sent it. */
  private accumulate(amount: number, from: Value): void {
    this.grad += amount;
    this.terms.push({ from, amount });
  }

  private static lift(x: Value | number): Value {
    return x instanceof Value ? x : new Value(x);
  }

  add(other: Value | number): Value {
    const o = Value.lift(other);
    const out = new Value(this.data + o.data, [this, o], '+');
    out.local = [1, 1];
    out._backward = () => {
      this.accumulate(1 * out.grad, out);
      o.accumulate(1 * out.grad, out);
    };
    return out;
  }

  mul(other: Value | number): Value {
    const o = Value.lift(other);
    const out = new Value(this.data * o.data, [this, o], '*');
    out.local = [o.data, this.data];
    out._backward = () => {
      this.accumulate(o.data * out.grad, out);
      o.accumulate(this.data * out.grad, out);
    };
    return out;
  }

  pow(k: number): Value {
    const out = new Value(this.data ** k, [this], `**${k}`);
    out.local = [k * this.data ** (k - 1)];
    out._backward = () => {
      this.accumulate(k * this.data ** (k - 1) * out.grad, out);
    };
    return out;
  }

  relu(): Value {
    const out = new Value(Math.max(this.data, 0), [this], 'relu');
    out.local = [this.data > 0 ? 1 : 0];
    out._backward = () => {
      this.accumulate((this.data > 0 ? 1 : 0) * out.grad, out);
    };
    return out;
  }

  /** Not in the app's network; kept for the ReLU-vs-tanh reliability measurement. */
  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], 'tanh');
    out.local = [1 - t * t];
    out._backward = () => {
      this.accumulate((1 - t * t) * out.grad, out);
    };
    return out;
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value | number): Value {
    return this.add(Value.lift(other).neg());
  }

  /**
   * Reverse-mode autodiff from this node.
   * Returns the order in which nodes were processed (reverse topological),
   * so a view can replay the pass in the engine's true order.
   */
  backward(): Value[] {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build = (v: Value): void => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v.children) build(child);
      topo.push(v);
    };
    build(this);

    for (const v of topo) {
      v.grad = 0;
      v.terms = [];
    }
    this.grad = 1;
    const order = topo.reverse();
    for (const node of order) node._backward();
    return order;
  }
}
