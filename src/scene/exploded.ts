/**
 * One neuron opened into its computation graph: a multiply node per input, a sum, a ReLU gate,
 * and the branches to every consumer of its output. The backward pass is replayed node by node.
 *
 * Simplification (stated in the article): the engine adds with a chain of two-input `+` nodes;
 * here they are drawn as one sum node, because every one of them just passes the gradient along.
 */
import {
  AdditiveBlending,
  BoxGeometry,
  Color,
  CylinderGeometry,
  Group,
  IcosahedronGeometry,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  OctahedronGeometry,
  Quaternion,
  SphereGeometry,
  TorusGeometry,
  Vector3,
  type Camera,
} from 'three';
import { SIZES } from '../engine/params';
import type { Snapshot } from '../engine/trace';
import { fmt } from '../model/format';
import { consumersOf } from '../model/store';
import type { Labels } from './labels';
import { COLORS, ease, NET_Y } from './theme';

const UP = new Vector3(0, 1, 0);
const q = new Quaternion();
const d = new Vector3();
const cPos = new Color(COLORS.pos);
const cNeg = new Color(COLORS.neg);
const cFwd = new Color(COLORS.fwd);
const cBack = new Color(COLORS.back);

type Edge = { mesh: Mesh; sheath: Mesh; sheathMat: MeshBasicMaterial; a: Vector3; b: Vector3 };

const PULSE_T = 0.7;

/** What the engine recorded for one neuron, arranged for this view. */
export type NeuronFacts = {
  l: number;
  j: number;
  nIn: number;
  m: number;
  isOutput: boolean;
  aIn: number[];
  w: number[];
  b: number;
  prod: number[];
  z: number;
  a: number;
  gate: number;
  /** `+=` terms received by this neuron's output, in the order the engine applied them. */
  terms: { j: number; w: number; delta: number; amount: number }[];
  dA: number;
  delta: number;
  dProd: number[];
  dW: number[];
  dB: number;
  /** What this neuron sends back to each of its inputs: w_k * delta. */
  toInput: number[];
};

export function neuronFacts(snap: Snapshot, l: number, j: number): NeuronFacts {
  const isOutput = l === 2;
  const nIn = SIZES[l];
  const terms = isOutput
    ? [{ j: 0, w: 1, delta: snap.dLossdDiff, amount: snap.dA[3][0] }]
    : snap.aTerms[l + 1][j];
  return {
    l,
    j,
    nIn,
    m: consumersOf(l),
    isOutput,
    aIn: snap.a[l],
    w: snap.params.W[l][j],
    b: snap.params.b[l][j],
    prod: snap.prod[l][j],
    z: snap.z[l][j],
    a: snap.a[l + 1][j],
    gate: snap.dAct[l][j],
    terms,
    dA: snap.dA[l + 1][j],
    delta: snap.delta[l][j],
    dProd: snap.dProd[l][j],
    dW: snap.grads.W[l][j],
    dB: snap.grads.b[l][j],
    toInput: Array.from({ length: nIn }, (_, k) => snap.aTerms[l][k].find((t) => t.j === j)!.amount),
  };
}

export type ExStage = { arrivals: number; gate: boolean; sum: boolean; mul: boolean; kind: 'forward' | 'arrive' | 'gate' | 'sum' | 'mul' };

export function exStage(step: number, m: number): ExStage {
  return {
    arrivals: Math.max(0, Math.min(m, step)),
    gate: step >= m + 1,
    sum: step >= m + 2,
    mul: step >= m + 3,
    kind: step === 0 ? 'forward' : step <= m ? 'arrive' : step === m + 1 ? 'gate' : step === m + 2 ? 'sum' : 'mul',
  };
}

function neuronName(l: number, j: number): string {
  return l === 2 ? 'the output neuron' : `hidden ${l + 1}, neuron ${j + 1}`;
}

/** The live sentence for each sub-step. Every number is read from the snapshot. */
export function explodedSay(f: NeuronFacts, step: number): string {
  const st = exStage(step, f.m);
  const name = neuronName(f.l, f.j);
  if (st.kind === 'forward') {
    const sum = f.prod.map((p) => fmt(p)).join(' + ');
    return `Inside ${name}: each input is multiplied by its weight, the products and the bias are added, z = ${sum} + ${fmt(f.b)} = <i>${fmt(f.z)}</i>${
      f.isOutput ? ', and the output layer is linear so ŷ = z.' : `, and ReLU gives a = <i>${fmt(f.a)}</i>.`
    } Press Next to run the backward pass through it.`;
  }
  if (st.kind === 'arrive') {
    const t = f.terms[step - 1];
    const sofar = f.terms.slice(0, step).reduce((s, x) => s + x.amount, 0);
    if (f.isOutput) {
      return `The loss sends back its slope: ∂L/∂ŷ = 2(ŷ − y) = <i>${fmt(t.amount)}</i>. The output has one consumer, so that is its whole gradient.`;
    }
    const more = step < f.m ? `${f.m - step} more to come: the gradient is not complete until every consumer has reported.` : 'All consumers have reported.';
    return `Consumer ${step} of ${f.m} (next-layer neuron ${t.j + 1}) sends back w × δ = ${fmt(t.w)} × ${fmt(t.delta)} = <i>${fmt(t.amount)}</i>. It is added with +=, total so far <i>${fmt(sofar)}</i>. ${more}`;
  }
  if (st.kind === 'gate') {
    if (f.isOutput) return `No ReLU on the output layer, so the gradient passes straight through: δ = <i>${fmt(f.delta)}</i>.`;
    return f.gate === 1
      ? `ReLU is a gate. z = ${fmt(f.z)} was positive, so the gate is open and multiplies by 1: δ = 1 × ${fmt(f.dA)} = <i>${fmt(f.delta)}</i>.`
      : `ReLU is a gate. z = ${fmt(f.z)} was not positive, so the gate is closed and multiplies by 0: δ = 0 × ${fmt(f.dA)} = <i>${fmt(f.delta)}</i>. Nothing behind this neuron learns from this sample.`;
  }
  if (st.kind === 'sum') {
    return `Addition distributes: every input of the sum receives δ = <i>${fmt(f.delta)}</i> unchanged. That is why ∂L/∂b = δ = <i>${fmt(f.dB)}</i>.`;
  }
  const k = 0;
  return `Multiplication swaps: each side gets δ times the other side. Top row: ∂L/∂w = a × δ = ${fmt(f.aIn[k])} × ${fmt(f.delta)} = <i>${fmt(f.dW[k])}</i>, and the input gets w × δ = ${fmt(f.w[k])} × ${fmt(f.delta)} = <i>${fmt(f.toInput[k])}</i>, one += for the neuron before.`;
}

export class ExplodedView {
  readonly group = new Group();
  private nodes: Mesh[] = [];
  private aNodes: Mesh[] = [];
  private wNodes: Mesh[] = [];
  private mulNodes: Mesh[] = [];
  private consNodes: Mesh[] = [];
  private sumNode: Mesh;
  private bNode: Mesh;
  private gate: Mesh;
  private gateBar: Mesh;
  private outNode: Mesh;
  private edges = new Map<string, Edge>();
  private pulses: Mesh[] = [];
  private pos = new Map<string, Vector3>();
  private world = new Vector3();

  constructor() {
    this.group.position.set(0.55, NET_Y + 0.05, 3.3);
    this.group.scale.setScalar(0.72);
    this.group.visible = false;

    const sphere = new SphereGeometry(1, 28, 20);
    const box = new BoxGeometry(1, 1, 1);
    const std = (c: number, e = 0.5): MeshStandardMaterial =>
      new MeshStandardMaterial({ color: new Color(c).multiplyScalar(0.5), emissive: new Color(c).multiplyScalar(e), roughness: 0.35, metalness: 0.2 });
    const add = (geo: SphereGeometry | BoxGeometry | OctahedronGeometry | IcosahedronGeometry, mat: MeshStandardMaterial, s: number): Mesh => {
      const m = new Mesh(geo, mat);
      m.scale.setScalar(s);
      this.group.add(m);
      this.nodes.push(m);
      return m;
    };
    for (let k = 0; k < 3; k++) {
      this.aNodes.push(add(sphere, std(COLORS.fwd, 0.9), 0.26));
      this.wNodes.push(add(box, std(COLORS.pos, 0.7), 0.3));
      this.mulNodes.push(add(new OctahedronGeometry(1, 0), std(0x8b97b8, 0.35), 0.3));
      this.consNodes.push(add(sphere, std(COLORS.back, 0.6), 0.2));
    }
    this.sumNode = add(new IcosahedronGeometry(1, 0), std(0x8b97b8, 0.35), 0.36);
    this.bNode = add(box, std(COLORS.pos, 0.7), 0.3);
    this.outNode = add(sphere, std(COLORS.fwd, 0.9), 0.32);
    this.gate = new Mesh(new TorusGeometry(0.42, 0.05, 12, 48), new MeshBasicMaterial({ color: COLORS.pos }));
    this.gate.rotation.y = Math.PI / 2 - 0.55;
    this.gateBar = new Mesh(new BoxGeometry(0.06, 0.95, 0.5), new MeshBasicMaterial({ color: new Color(COLORS.neg).multiplyScalar(1.3) }));
    this.group.add(this.gate, this.gateBar);

    const cyl = new CylinderGeometry(1, 1, 1, 10, 1, true);
    const wire = new MeshStandardMaterial({ color: 0x3b4868, emissive: 0x1c2438, roughness: 0.5 });
    const edge = (id: string): void => {
      const mesh = new Mesh(cyl, wire);
      const sheathMat = new MeshBasicMaterial({ color: cBack.clone(), transparent: true, opacity: 0, blending: AdditiveBlending, depthWrite: false });
      const sheath = new Mesh(cyl, sheathMat);
      this.group.add(mesh, sheath);
      this.edges.set(id, { mesh, sheath, sheathMat, a: new Vector3(), b: new Vector3() });
    };
    for (let k = 0; k < 3; k++) {
      edge(`a${k}`);
      edge(`w${k}`);
      edge(`m${k}`);
      edge(`c${k}`);
    }
    edge('b');
    edge('sg');
    edge('go');

    for (let i = 0; i < 8; i++) {
      const p = new Mesh(sphere, new MeshBasicMaterial({ color: cBack.clone().multiplyScalar(1.6) }));
      p.scale.setScalar(0.1);
      p.visible = false;
      this.group.add(p);
      this.pulses.push(p);
    }
  }

  private layout(f: NeuronFacts): void {
    const row = (k: number): number => ((f.nIn - 1) / 2 - k) * 1.9;
    const P = this.pos;
    for (let k = 0; k < 3; k++) {
      P.set(`a${k}`, new Vector3(-5.2, row(k) + 0.4, 0));
      P.set(`w${k}`, new Vector3(-5.2, row(k) - 0.4, 0));
      P.set(`m${k}`, new Vector3(-2.9, row(k), 0));
      P.set(`c${k}`, new Vector3(5.7, ((f.m - 1) / 2 - k) * 1.55, 0));
    }
    P.set('sum', new Vector3(-0.6, 0, 0));
    P.set('b', new Vector3(-0.6, f.nIn === 3 ? 2.35 : 1.9, 0));
    P.set('gate', new Vector3(1.6, 0, 0));
    P.set('out', new Vector3(3.6, 0, 0));
  }

  private setEdge(id: string, from: string, to: string, visible: boolean, lit: boolean): void {
    const e = this.edges.get(id)!;
    e.mesh.visible = visible;
    e.sheath.visible = visible && lit;
    if (!visible) return;
    e.a.copy(this.pos.get(from)!);
    e.b.copy(this.pos.get(to)!);
    for (const [mesh, r] of [
      [e.mesh, 0.022],
      [e.sheath, 0.055],
    ] as const) {
      d.subVectors(e.b, e.a);
      const len = d.length();
      mesh.position.addVectors(e.a, e.b).multiplyScalar(0.5);
      mesh.quaternion.copy(q.setFromUnitVectors(UP, d.normalize()));
      mesh.scale.set(r, len, r);
    }
    e.sheathMat.opacity = 0.55;
  }

  /** Update meshes for neuron facts `f` at sub-step `step`, `t` seconds after the sub-step began. */
  apply(f: NeuronFacts, step: number, t: number): void {
    this.layout(f);
    const st = exStage(step, f.m);
    const arrived = new Set(f.terms.slice(0, st.arrivals).map((x) => x.j));

    for (let k = 0; k < 3; k++) {
      const on = k < f.nIn;
      this.aNodes[k].visible = this.wNodes[k].visible = this.mulNodes[k].visible = on;
      if (on) {
        this.aNodes[k].position.copy(this.pos.get(`a${k}`)!);
        this.wNodes[k].position.copy(this.pos.get(`w${k}`)!);
        this.mulNodes[k].position.copy(this.pos.get(`m${k}`)!);
        const wm = this.wNodes[k].material as MeshStandardMaterial;
        const c = f.w[k] >= 0 ? cPos : cNeg;
        wm.color.copy(c).multiplyScalar(0.5);
        wm.emissive.copy(c).multiplyScalar(0.7);
        const am = this.aNodes[k].material as MeshStandardMaterial;
        am.emissive.copy(cFwd).multiplyScalar(f.aIn[k] === 0 ? 0.12 : 0.9);
      }
      this.setEdge(`a${k}`, `a${k}`, `m${k}`, on, st.mul);
      this.setEdge(`w${k}`, `w${k}`, `m${k}`, on, st.mul);
      this.setEdge(`m${k}`, `m${k}`, 'sum', on, st.sum);
      const con = k < f.m;
      this.consNodes[k].visible = con;
      if (con) this.consNodes[k].position.copy(this.pos.get(`c${k}`)!);
      this.setEdge(`c${k}`, 'out', `c${k}`, con, arrived.has(k));
    }
    this.sumNode.position.copy(this.pos.get('sum')!);
    this.bNode.position.copy(this.pos.get('b')!);
    const bm = this.bNode.material as MeshStandardMaterial;
    const bc = f.b >= 0 ? cPos : cNeg;
    bm.color.copy(bc).multiplyScalar(0.5);
    bm.emissive.copy(bc).multiplyScalar(0.7);
    this.outNode.position.copy(this.pos.get('out')!);
    (this.outNode.material as MeshStandardMaterial).emissive.copy(cFwd).multiplyScalar(f.a === 0 ? 0.12 : 0.9);
    this.gate.position.copy(this.pos.get('gate')!);
    this.gateBar.position.copy(this.pos.get('gate')!);
    const closed = !f.isOutput && f.gate === 0;
    this.gateBar.visible = closed;
    (this.gate.material as MeshBasicMaterial).color.copy(closed ? cNeg : cPos).multiplyScalar(f.isOutput ? 0.6 : 1.3);
    this.setEdge('b', 'b', 'sum', true, st.sum);
    this.setEdge('sg', 'sum', 'gate', true, st.gate);
    this.setEdge('go', 'gate', 'out', true, st.gate);

    // Pulses for the sub-step that just started, travelling against the forward direction.
    const routes: [string, string][] = [];
    if (t < PULSE_T) {
      if (st.kind === 'arrive') routes.push([`c${f.terms[step - 1].j}`, 'out']);
      if (st.kind === 'gate') routes.push(['out', 'sum']);
      if (st.kind === 'sum') {
        for (let k = 0; k < f.nIn; k++) routes.push(['sum', `m${k}`]);
        routes.push(['sum', 'b']);
      }
      if (st.kind === 'mul') {
        for (let k = 0; k < f.nIn; k++) routes.push([`m${k}`, `a${k}`], [`m${k}`, `w${k}`]);
      }
    }
    const u = ease(t / PULSE_T);
    this.pulses.forEach((p, i) => {
      const r = routes[i];
      p.visible = !!r;
      if (r) p.position.copy(this.pos.get(r[0])!).lerp(this.pos.get(r[1])!, u);
    });
  }

  private at(id: string, dx = 0, dy = 0): Vector3 {
    this.world.copy(this.pos.get(id)!);
    this.world.x += dx;
    this.world.y += dy;
    return this.group.localToWorld(this.world);
  }

  labels(L: Labels, cam: Camera, f: NeuronFacts, step: number): void {
    const st = exStage(step, f.m);
    const g = (s: string): string => `<span class="d">${s}</span>`;
    const inName = (k: number): string => (f.l === 0 ? `x<sub>${k + 1}</sub>` : `a<sub>${k + 1}</sub>`);
    for (let k = 0; k < f.nIn; k++) {
      L.put(
        `ex-a${k}`,
        this.at(`a${k}`, -0.4),
        `<span class="a">${inName(k)} = ${fmt(f.aIn[k])}</span>${st.mul ? `<br>${g(`grad += w·δ = ${fmt(f.toInput[k])}`)}` : ''}`,
        cam,
        'ex',
        'r',
      );
      L.put(
        `ex-w${k}`,
        this.at(`w${k}`, -0.4),
        `<span class="w">w<sub>${k + 1}</sub> = ${fmt(f.w[k])}</span>${st.mul ? `<br>${g(`∂L/∂w = a·δ = ${fmt(f.dW[k])}`)}` : ''}`,
        cam,
        'ex',
        'r',
      );
      L.put(
        `ex-m${k}`,
        this.at(`m${k}`, 0, -0.4),
        `<b>×</b> ${fmt(f.prod[k])}${st.sum ? `<br>${g(`grad = ${fmt(f.dProd[k])}`)}` : ''}`,
        cam,
        'ex',
        't',
      );
    }
    L.put(
      'ex-sum',
      this.at('sum', 0, -0.5),
      `<b>+</b> z = ${fmt(f.z)}${st.gate ? `<br>${g(`δ = ${fmt(f.delta)}`)}` : ''}`,
      cam,
      'ex',
      't',
    );
    L.put(
      'ex-b',
      this.at('b', 0, 0.3),
      `<span class="w">b = ${fmt(f.b)}</span>${st.sum ? `<br>${g(`∂L/∂b = δ = ${fmt(f.dB)}`)}` : ''}`,
      cam,
      'ex',
      'b',
    );
    L.put(
      'ex-gate',
      this.at('gate', 0, 0.6),
      f.isOutput ? 'linear<br>× 1' : `ReLU gate<br><b class="${f.gate ? 'w' : 'neg'}">${f.gate ? 'open × 1' : 'closed × 0'}</b>`,
      cam,
      'ex',
      'b',
    );
    const got = f.terms.slice(0, st.arrivals);
    const total = got.reduce((s, x) => s + x.amount, 0);
    const sumTxt = got.length > 1 ? `${got.map((x) => fmt(x.amount)).join(' + ')} = ${fmt(total)}` : got.length ? fmt(total) : '';
    L.put(
      'ex-out',
      this.at('out', 0, -0.5),
      `<span class="a">${f.isOutput ? 'ŷ' : 'a'} = ${fmt(f.a)}</span>${got.length ? `<br>${g(`grad = ${sumTxt}`)}` : ''}`,
      cam,
      'ex',
      't',
    );
    for (let i = 0; i < f.m; i++) {
      const term = f.terms.find((x) => x.j === i)!;
      const arrived = got.includes(term);
      const head = f.isOutput ? 'loss L' : `to neuron ${i + 1}<br>w ${fmt(term.w)} · δ ${fmt(term.delta)}`;
      const sends = f.isOutput ? `sends 2(ŷ − y) = ${fmt(term.amount)}` : `sends ${fmt(term.amount)}`;
      L.put(`ex-c${i}`, this.at(`c${i}`, 0.35), `${head}${arrived ? `<br>${g(sends)}` : ''}`, cam, 'ex', 'l');
    }
  }
}
