/**
 * The network as objects: neurons are spheres, weights are tubes (thickness = |w|, colour = sign),
 * biases are small satellites, and the loss is a physical gap next to the output.
 * This file only draws what it is handed in a NetFrame; it computes no numbers of its own.
 */
import {
  AdditiveBlending,
  BoxGeometry,
  Color,
  CylinderGeometry,
  DoubleSide,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  type Object3D,
  PlaneGeometry,
  Quaternion,
  SphereGeometry,
  TorusGeometry,
  Vector3,
} from 'three';
import { N_LAYERS, SIZES } from '../engine/params';
import type { Selection } from '../model/store';
import {
  biasRadius,
  COL_X,
  COLORS,
  GAUGE_MAX,
  GAUGE_MIN,
  GAUGE_SCALE,
  GAUGE_X,
  gaugeY,
  NET_Y,
  NEURON_R,
  neuronPos,
  tubeRadius,
} from './theme';

export type NetFrame = {
  /** Weights and biases as drawn (may be mid-way through the update animation). */
  w: number[][][];
  b: number[][];
  /** Activation per column and neuron, or null while it is not known yet. */
  act: (number | null)[][];
  /** Error signal per weight-layer and neuron, or null while it is not known yet. */
  delta: (number | null)[][];
  /** Gradient brightness per tube and per bias, 0..1 (0 = not revealed). */
  gradGlow: number[][][];
  biasGlow: number[][];
  /** 1 = normal, lower = the network steps back (Exploded view). */
  dim: number;
  selection: Selection;
  hover: Selection;
  yhat: number | null;
  target: number | null;
  /** 0..1 growth of the loss gap. */
  gap: number;
  /** Verify: a visible wobble of one parameter, in display units. */
  twitch: { sel: Selection; amount: number } | null;
};

const UP = new Vector3(0, 1, 0);
const tmpDir = new Vector3();
const tmpQ = new Quaternion();
const tmpC = new Color();

const cPos = new Color(COLORS.pos);
const cNeg = new Color(COLORS.neg);
const cFwd = new Color(COLORS.fwd);
const cBack = new Color(COLORS.back);
const cTarget = new Color(COLORS.target);
const cIdle = new Color(0x2a3550);
const cShell = new Color(0x1c3a52);

type Tube = {
  mesh: Mesh;
  mat: MeshStandardMaterial;
  sheath: Mesh;
  sheathMat: MeshBasicMaterial;
  from: Vector3;
  to: Vector3;
  mid: Vector3;
  len: number;
};

type Neuron = { mesh: Mesh; mat: MeshStandardMaterial; halo: Mesh; haloMat: MeshBasicMaterial; pos: Vector3 };
type Bias = { mesh: Mesh; mat: MeshStandardMaterial; pos: Vector3 };

export function sameSel(a: Selection, b: Selection): boolean {
  if (!a || !b) return a === b;
  return JSON.stringify(a) === JSON.stringify(b);
}

function orient(mesh: Object3D, from: Vector3, to: Vector3, radius: number): void {
  tmpDir.subVectors(to, from);
  const len = tmpDir.length();
  mesh.position.addVectors(from, to).multiplyScalar(0.5);
  mesh.quaternion.copy(tmpQ.setFromUnitVectors(UP, tmpDir.normalize()));
  mesh.scale.set(radius, len, radius);
}

export class NetworkView {
  readonly group = new Group();
  readonly neurons: Neuron[][] = [];
  readonly tubes: Tube[][][] = [];
  readonly biases: Bias[][] = [];
  /** Meshes the raycaster tests; each carries its Selection in userData.sel. */
  readonly pickables: Object3D[] = [];
  private ring: Mesh;
  private gauge = new Group();
  private targetRing: Mesh;
  private yhatDot: Mesh;
  private yhatLink: Mesh;
  private gapBar: Mesh;
  private square: Mesh;
  private squareEdge: Mesh[] = [];

  constructor() {
    const sphere = new SphereGeometry(1, 40, 28);
    const cyl = new CylinderGeometry(1, 1, 1, 16, 1, true);
    const haloGeo = new TorusGeometry(NEURON_R + 0.12, 0.022, 10, 56);

    for (let col = 0; col < SIZES.length; col++) {
      const column: Neuron[] = [];
      for (let i = 0; i < SIZES[col]; i++) {
        const pos = neuronPos(col, i);
        const mat = new MeshStandardMaterial({ color: cShell.clone(), roughness: 0.28, metalness: 0.35, emissive: cIdle.clone() });
        const mesh = new Mesh(sphere, mat);
        const r = col === 0 ? NEURON_R * 0.82 : NEURON_R;
        mesh.scale.setScalar(r);
        mesh.position.copy(pos);
        mesh.userData.sel = { kind: 'neuron', col, i } satisfies Selection;
        this.group.add(mesh);
        this.pickables.push(mesh);

        const haloMat = new MeshBasicMaterial({ color: cBack.clone(), transparent: true, opacity: 0, depthWrite: false });
        const halo = new Mesh(haloGeo, haloMat);
        halo.position.copy(pos);
        halo.visible = false;
        this.group.add(halo);
        column.push({ mesh, mat, halo, haloMat, pos });
      }
      this.neurons.push(column);
    }

    const pickMat = new MeshBasicMaterial({ visible: false });
    for (let l = 0; l < N_LAYERS; l++) {
      const layer: Tube[][] = [];
      const biasRow: Bias[] = [];
      for (let j = 0; j < SIZES[l + 1]; j++) {
        const row: Tube[] = [];
        const dst = this.neurons[l + 1][j].pos;
        for (let k = 0; k < SIZES[l]; k++) {
          const src = this.neurons[l][k].pos;
          tmpDir.subVectors(dst, src).normalize();
          const from = src.clone().addScaledVector(tmpDir, NEURON_R * 0.86);
          const to = dst.clone().addScaledVector(tmpDir, -NEURON_R * 0.92);
          const mat = new MeshStandardMaterial({ color: cPos.clone(), emissive: cPos.clone(), roughness: 0.4, metalness: 0.1 });
          const mesh = new Mesh(cyl, mat);
          const sheathMat = new MeshBasicMaterial({
            color: cBack.clone(),
            transparent: true,
            opacity: 0,
            blending: AdditiveBlending,
            depthWrite: false,
          });
          const sheath = new Mesh(cyl, sheathMat);
          sheath.visible = false;
          const pick = new Mesh(cyl, pickMat);
          orient(pick, from, to, 0.11);
          pick.userData.sel = { kind: 'weight', l, j, k } satisfies Selection;
          this.group.add(mesh, sheath, pick);
          this.pickables.push(pick);
          row.push({ mesh, mat, sheath, sheathMat, from, to, mid: from.clone().lerp(to, 0.5), len: from.distanceTo(to) });
        }
        layer.push(row);

        // Bias: a satellite up and to the left of its neuron, on a short stalk.
        const pos = dst.clone().add(new Vector3(-0.36, 0.5, 0.22));
        const mat = new MeshStandardMaterial({ color: cPos.clone(), emissive: cPos.clone(), roughness: 0.35 });
        const mesh = new Mesh(sphere, mat);
        mesh.position.copy(pos);
        mesh.userData.sel = { kind: 'bias', l, j } satisfies Selection;
        const stalk = new Mesh(cyl, new MeshStandardMaterial({ color: 0x3a4664, roughness: 0.5 }));
        orient(stalk, dst.clone().add(new Vector3(-0.2, 0.3, 0.12)), pos, 0.012);
        const bpick = new Mesh(sphere, pickMat);
        bpick.position.copy(pos);
        bpick.scale.setScalar(0.19);
        bpick.userData.sel = mesh.userData.sel;
        this.group.add(mesh, stalk, bpick);
        this.pickables.push(bpick);
        biasRow.push({ mesh, mat, pos });
      }
      this.tubes.push(layer);
      this.biases.push(biasRow);
    }

    // Selection ring (billboarded each frame).
    this.ring = new Mesh(
      new TorusGeometry(1, 0.022, 8, 64),
      new MeshBasicMaterial({ color: 0xaab6d6, transparent: true, opacity: 0.85, depthTest: false }),
    );
    this.ring.visible = false;
    this.ring.renderOrder = 5;
    this.group.add(this.ring);

    // Loss gauge: a rail with the prediction and the target on it. The gap between them is the error.
    const railMat = new MeshStandardMaterial({ color: 0x2b3653, roughness: 0.5, metalness: 0.6 });
    const rail = new Mesh(new BoxGeometry(0.035, (GAUGE_MAX - GAUGE_MIN) * GAUGE_SCALE, 0.035), railMat);
    rail.position.set(GAUGE_X, NET_Y + ((GAUGE_MAX + GAUGE_MIN) / 2 - 0.5) * GAUGE_SCALE, 0);
    this.gauge.add(rail);
    for (const tick of [0, 1]) {
      const t = new Mesh(new BoxGeometry(0.22, 0.02, 0.04), railMat);
      t.position.set(GAUGE_X - 0.1, gaugeY(tick), 0);
      this.gauge.add(t);
    }
    this.targetRing = new Mesh(
      new TorusGeometry(0.15, 0.03, 10, 40),
      new MeshBasicMaterial({ color: cTarget.clone().multiplyScalar(1.4) }),
    );
    this.yhatDot = new Mesh(new SphereGeometry(0.1, 20, 14), new MeshBasicMaterial({ color: cFwd.clone().multiplyScalar(1.5) }));
    this.yhatLink = new Mesh(cyl, new MeshBasicMaterial({ color: cFwd.clone().multiplyScalar(0.55) }));
    this.gapBar = new Mesh(new BoxGeometry(0.06, 1, 0.06), new MeshBasicMaterial({ color: cTarget.clone().multiplyScalar(1.5) }));
    this.square = new Mesh(
      new PlaneGeometry(1, 1),
      new MeshBasicMaterial({ color: cTarget, transparent: true, opacity: 0.2, side: DoubleSide, depthWrite: false }),
    );
    const edgeMat = new MeshBasicMaterial({ color: cTarget.clone().multiplyScalar(1.1) });
    for (let i = 0; i < 4; i++) {
      const e = new Mesh(new BoxGeometry(1, 1, 1), edgeMat);
      this.squareEdge.push(e);
      this.gauge.add(e);
    }
    this.gauge.add(this.targetRing, this.yhatDot, this.yhatLink, this.gapBar, this.square);
    this.group.add(this.gauge);
  }

  /** Where a label for this selection should sit. */
  anchorOf(sel: Selection): Vector3 | null {
    if (!sel) return null;
    if (sel.kind === 'neuron') return this.neurons[sel.col][sel.i].pos;
    if (sel.kind === 'weight') return this.tubes[sel.l][sel.j][sel.k].mid;
    return this.biases[sel.l][sel.j].pos;
  }

  /** Point at fraction u along a tube (0 = source side). */
  along(l: number, j: number, k: number, u: number, out: Vector3): Vector3 {
    const t = this.tubes[l][j][k];
    return out.copy(t.from).lerp(t.to, u);
  }

  apply(f: NetFrame, cameraPos: Vector3): void {
    const dim = f.dim;

    for (let col = 0; col < SIZES.length; col++) {
      for (let i = 0; i < SIZES[col]; i++) {
        const n = this.neurons[col][i];
        const a = f.act[col][i];
        const sel: Selection = { kind: 'neuron', col, i };
        const hot = sameSel(f.hover, sel) ? 0.25 : 0;
        if (a === null) {
          n.mat.emissive.copy(cIdle).multiplyScalar((0.35 + hot) * dim);
        } else {
          // Brightness follows |a|; a value of exactly 0 stays visibly dark ("off").
          const level = a === 0 ? 0.1 : 0.3 + 0.55 * Math.min(1, Math.abs(a) / 1.5);
          n.mat.emissive.copy(cFwd).multiplyScalar((level + hot) * dim);
        }
        n.mat.color.copy(cShell).multiplyScalar(dim);

        const d = col >= 1 ? f.delta[col - 1][i] : null;
        if (d === null || dim < 0.9) {
          n.halo.visible = false;
        } else {
          n.halo.visible = true;
          n.halo.quaternion.copy(tmpQ.setFromUnitVectors(new Vector3(0, 0, 1), tmpDir.subVectors(cameraPos, n.pos).normalize()));
          if (d === 0) {
            n.haloMat.color.setHex(0x5a6788);
            n.haloMat.opacity = 0.7;
          } else {
            n.haloMat.color.copy(cBack).multiplyScalar(1.15);
            n.haloMat.opacity = 0.95;
          }
        }
      }
    }

    for (let l = 0; l < N_LAYERS; l++) {
      for (let j = 0; j < SIZES[l + 1]; j++) {
        for (let k = 0; k < SIZES[l]; k++) {
          const t = this.tubes[l][j][k];
          const sel: Selection = { kind: 'weight', l, j, k };
          let w = f.w[l][j][k];
          if (f.twitch && sameSel(f.twitch.sel, sel)) w += f.twitch.amount;
          const selected = sameSel(f.selection, sel);
          const hovered = sameSel(f.hover, sel);
          const r = tubeRadius(w) + (selected ? 0.012 : 0);
          orient(t.mesh, t.from, t.to, r);
          tmpC.copy(w >= 0 ? cPos : cNeg);
          t.mat.color.copy(tmpC).multiplyScalar(0.55 * dim);
          t.mat.emissive.copy(tmpC).multiplyScalar((selected ? 1.5 : hovered ? 1.0 : 0.5) * dim);
          const g = f.gradGlow[l][j][k];
          if (g > 0 && dim > 0.9) {
            t.sheath.visible = true;
            orient(t.sheath, t.from, t.to, r + 0.035 + 0.045 * g);
            t.sheathMat.opacity = 0.07 + 0.5 * g;
          } else {
            t.sheath.visible = false;
          }
        }

        const bias = this.biases[l][j];
        const bsel: Selection = { kind: 'bias', l, j };
        let b = f.b[l][j];
        if (f.twitch && sameSel(f.twitch.sel, bsel)) b += f.twitch.amount;
        bias.mesh.scale.setScalar(biasRadius(b) * (sameSel(f.selection, bsel) ? 1.25 : 1));
        tmpC.copy(b >= 0 ? cPos : cNeg);
        const bg = f.biasGlow[l][j];
        bias.mat.color.copy(tmpC).multiplyScalar(0.5 * dim);
        bias.mat.emissive
          .copy(tmpC)
          .multiplyScalar(sameSel(f.hover, bsel) ? 1.1 : 0.6)
          .lerp(tmpC.copy(cBack).multiplyScalar(1.6), bg > 0 ? 0.35 + 0.65 * bg : 0)
          .multiplyScalar(dim);
      }
    }

    // Selection ring.
    const s = f.selection;
    if (s && s.kind !== 'weight' && dim > 0.9) {
      const p = this.anchorOf(s)!;
      this.ring.visible = true;
      this.ring.position.copy(p);
      this.ring.scale.setScalar(s.kind === 'neuron' ? NEURON_R + 0.22 : 0.24);
      this.ring.quaternion.copy(tmpQ.setFromUnitVectors(new Vector3(0, 0, 1), tmpDir.subVectors(cameraPos, p).normalize()));
    } else {
      this.ring.visible = false;
    }

    // Loss gauge.
    this.gauge.visible = dim > 0.9;
    this.targetRing.visible = f.target !== null;
    if (f.target !== null) this.targetRing.position.set(GAUGE_X, gaugeY(f.target), 0);
    const showY = f.yhat !== null;
    this.yhatDot.visible = showY;
    this.yhatLink.visible = showY;
    const showGap = showY && f.target !== null && f.gap > 0.001;
    this.gapBar.visible = showGap;
    this.square.visible = showGap;
    for (const e of this.squareEdge) e.visible = showGap;
    if (showY) {
      const yy = gaugeY(f.yhat!);
      this.yhatDot.position.set(GAUGE_X, yy, 0);
      const out = this.neurons[3][0].pos;
      orient(this.yhatLink, new Vector3(out.x + NEURON_R, out.y, 0), new Vector3(GAUGE_X, yy, 0), 0.012);
      if (showGap) {
        const ty = gaugeY(f.target!);
        // The bar grows from the prediction towards the target; the square's side is the same length.
        const end = yy + (ty - yy) * f.gap;
        const side = Math.max(1e-4, Math.abs(end - yy));
        this.gapBar.position.set(GAUGE_X, (yy + end) / 2, 0);
        this.gapBar.scale.set(1, side, 1);
        const lo = Math.min(yy, end);
        const x0 = GAUGE_X + 0.14;
        this.square.position.set(x0 + side / 2, lo + side / 2, 0);
        this.square.scale.set(side, side, 1);
        const th = 0.018;
        this.squareEdge[0].position.set(x0 + side / 2, lo, 0);
        this.squareEdge[0].scale.set(side, th, th);
        this.squareEdge[1].position.set(x0 + side / 2, lo + side, 0);
        this.squareEdge[1].scale.set(side, th, th);
        this.squareEdge[2].position.set(x0, lo + side / 2, 0);
        this.squareEdge[2].scale.set(th, side, th);
        this.squareEdge[3].position.set(x0 + side, lo + side / 2, 0);
        this.squareEdge[3].scale.set(th, side, th);
      }
    }
  }

  /** Centre of the loss square, for its label. */
  squareCentre(yhat: number, target: number): Vector3 {
    const a = gaugeY(yhat);
    const b = gaugeY(target);
    const side = Math.abs(a - b);
    return new Vector3(GAUGE_X + 0.14 + side / 2, Math.min(a, b) + side / 2, 0);
  }
}

export { COL_X };
