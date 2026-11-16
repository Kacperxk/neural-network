/**
 * Descent view: a second, small 3D scene. The loss as a surface over two chosen parameters
 * (the other 23 frozen at their current values), a ball at the current point, the gradient arrow,
 * and where one step of size lr would land.
 *
 * The surface is a grid of plain forward passes (matrix.ts, tested equal to the engine).
 * The ball's loss and the gradient come from the engine via the store.
 */
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  ConeGeometry,
  CylinderGeometry,
  DirectionalLight,
  DoubleSide,
  Group,
  HemisphereLight,
  Line,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PerspectiveCamera,
  Quaternion,
  Scene,
  SphereGeometry,
  Vector3,
  WebGLRenderer,
} from 'three';
import { batchLoss, sampleLoss } from '../engine/matrix';
import { allRefs, cloneParams, getParam, refKey, setParam, type ParamRef, type Params } from '../engine/params';
import { fmt } from '../model/format';
import type { Store } from '../model/store';
import { COLORS } from './theme';

const N = 41;
/** Half-width of the window in parameter units. */
const RANGE = 1.5;
/** Half-width of the surface in scene units. */
const HALF = 2;
const TRAIL = 80;

const height = (loss: number): number => 0.9 * Math.log1p(2 * Math.max(0, loss));

export function paramLabel(ref: ParamRef): string {
  const L = ref.l + 1;
  const dst = ref.l === 2 ? 'ŷ' : `h${L}${ref.j + 1}`;
  if (ref.kind === 'b') return `b${L} · ${dst}`;
  const src = ref.l === 0 ? `x${ref.k + 1}` : `h${ref.l}${ref.k + 1}`;
  return `w${L} · ${src}→${dst}`;
}

const UP = new Vector3(0, 1, 0);

class Arrow {
  readonly group = new Group();
  private shaft: Mesh;
  private head: Mesh;
  constructor(color: number) {
    const mat = new MeshBasicMaterial({ color });
    this.shaft = new Mesh(new CylinderGeometry(0.025, 0.025, 1, 10), mat);
    this.head = new Mesh(new ConeGeometry(0.075, 0.2, 14), mat);
    this.group.add(this.shaft, this.head);
  }
  set(from: Vector3, to: Vector3): void {
    const d = new Vector3().subVectors(to, from);
    const len = d.length();
    this.group.visible = len > 0.02;
    if (!this.group.visible) return;
    const q = new Quaternion().setFromUnitVectors(UP, d.clone().normalize());
    const headLen = Math.min(0.2, len * 0.5);
    this.shaft.position.copy(from).addScaledVector(d, (len - headLen) / (2 * len));
    this.shaft.quaternion.copy(q);
    this.shaft.scale.set(1, len - headLen, 1);
    this.head.position.copy(to).addScaledVector(d, -headLen / (2 * len));
    this.head.quaternion.copy(q);
    this.head.scale.set(1, headLen / 0.2, 1);
  }
}

export class DescentView {
  private renderer: WebGLRenderer;
  private scene = new Scene();
  private camera = new PerspectiveCamera(36, 1, 0.1, 50);
  private geo = new BufferGeometry();
  private wire: LineSegments;
  private ball: Mesh;
  private ghost: Mesh;
  private gradArrow = new Arrow(COLORS.back);
  private stepArrow = new Arrow(COLORS.fwd);
  private trailLine: Line;
  private trail: Vector3[] = [];
  private centre: [number, number] = [0, 0];
  private key = '';
  private needCentre = true;
  private dirty = true;
  private az = -0.7;
  private el = 0.62;
  private dragging = false;
  private lastParams: Params | null = null;
  /** Text for the tile's footer. */
  footer = '';

  constructor(
    private canvas: HTMLCanvasElement,
    private store: Store,
  ) {
    this.renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x000000, 0);

    const pos = new Float32Array(N * N * 3);
    const col = new Float32Array(N * N * 3);
    const idx: number[] = [];
    for (let i = 0; i < N - 1; i++) {
      for (let j = 0; j < N - 1; j++) {
        const a = i * N + j;
        idx.push(a, a + N, a + 1, a + 1, a + N, a + N + 1);
      }
    }
    this.geo.setAttribute('position', new BufferAttribute(pos, 3));
    this.geo.setAttribute('color', new BufferAttribute(col, 3));
    this.geo.setIndex(idx);
    const surface = new Mesh(this.geo, new MeshStandardMaterial({ vertexColors: true, side: DoubleSide, roughness: 0.55, metalness: 0.1 }));
    this.scene.add(surface);

    // Every fifth grid line, drawn on top, so the shape reads without relying on shading alone.
    const wireIdx: number[] = [];
    for (let i = 0; i < N; i += 5) {
      for (let j = 0; j < N - 1; j++) {
        wireIdx.push(i * N + j, i * N + j + 1, j * N + i, (j + 1) * N + i);
      }
    }
    const wireGeo = new BufferGeometry();
    wireGeo.setAttribute('position', this.geo.getAttribute('position'));
    wireGeo.setIndex(wireIdx);
    this.wire = new LineSegments(wireGeo, new LineBasicMaterial({ color: 0xdfe8ff, transparent: true, opacity: 0.22 }));
    this.wire.position.y = 0.006;
    this.scene.add(this.wire);

    this.ball = new Mesh(new SphereGeometry(0.11, 24, 16), new MeshBasicMaterial({ color: 0xffffff }));
    this.ghost = new Mesh(new SphereGeometry(0.085, 20, 14), new MeshBasicMaterial({ color: COLORS.fwd, transparent: true, opacity: 0.75 }));
    this.trailLine = new Line(new BufferGeometry(), new LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 }));
    this.scene.add(this.ball, this.ghost, this.gradArrow.group, this.stepArrow.group, this.trailLine);

    // Axis markers: the two parameter directions.
    const axis = (color: number, a: Vector3, b: Vector3): void => {
      const arrow = new Arrow(color);
      arrow.set(a, b);
      this.scene.add(arrow.group);
    };
    axis(COLORS.pos, new Vector3(-HALF, 0, HALF + 0.15), new Vector3(-HALF + 1.1, 0, HALF + 0.15));
    axis(COLORS.target, new Vector3(-HALF - 0.15, 0, HALF), new Vector3(-HALF - 0.15, 0, HALF - 1.1));

    this.scene.add(new HemisphereLight(0xcfe0ff, 0x10131c, 1.5));
    const sun = new DirectionalLight(0xffffff, 1.6);
    sun.position.set(3, 6, 2);
    this.scene.add(sun);

    canvas.addEventListener('pointerdown', (e) => {
      this.dragging = true;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      this.az -= e.movementX * 0.01;
      this.el = Math.max(0.15, Math.min(1.35, this.el + e.movementY * 0.008));
      this.dirty = true;
    });
    const stop = (): void => {
      this.dragging = false;
    };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);

    store.subscribe((s, prev) => {
      this.dirty = true;
      // The weights were reloaded (new seed or Reset): start a fresh map.
      if (s.seed !== prev.seed || (s.train.iter === 0 && prev.train.iter > 0) || (s.iteration === 0 && prev.iteration > 0)) this.needCentre = true;
    });
    new ResizeObserver(() => {
      this.dirty = true;
    }).observe(canvas);
  }

  private lossFn(): (p: Params) => number {
    const s = this.store.s;
    return s.mode === 'train' ? (p) => batchLoss(p) : (p) => sampleLoss(p, s.sample);
  }

  private toScene(a: number, b: number, loss: number, out: Vector3): Vector3 {
    return out.set(((a - this.centre[0]) / RANGE) * HALF, height(loss), (-(b - this.centre[1]) / RANGE) * HALF);
  }

  private rebuild(): void {
    const store = this.store;
    const s = store.s;
    const { a: refA, b: refB } = s.descent;
    const a0 = getParam(s.params, refA);
    const b0 = getParam(s.params, refB);

    // Recentre when the axes, the loss being shown or the starting weights change, or when the ball leaves the map.
    const key = `${refKey(refA)}|${refKey(refB)}|${s.mode}|${s.mode === 'step' ? s.sample : ''}`;
    const outside = Math.abs(a0 - this.centre[0]) > RANGE * 0.9 || Math.abs(b0 - this.centre[1]) > RANGE * 0.9;
    if (key !== this.key || outside || this.needCentre) {
      this.centre = [a0, b0];
      this.trail = [];
    }
    this.key = key;
    this.needCentre = false;

    const loss = this.lossFn();
    const work = cloneParams(s.params);
    const pos = this.geo.getAttribute('position') as BufferAttribute;
    const col = this.geo.getAttribute('color') as BufferAttribute;
    const lo = new Color(0x1a2a5a);
    const mid = new Color(COLORS.fwd).multiplyScalar(0.75);
    const hi = new Color(0xf4f7ff);
    const c = new Color();
    const v = new Vector3();
    for (let i = 0; i < N; i++) {
      const a = this.centre[0] + ((i / (N - 1)) * 2 - 1) * RANGE;
      setParam(work, refA, a);
      for (let j = 0; j < N; j++) {
        const b = this.centre[1] + ((j / (N - 1)) * 2 - 1) * RANGE;
        setParam(work, refB, b);
        const L = loss(work);
        this.toScene(a, b, L, v);
        const n = i * N + j;
        pos.setXYZ(n, v.x, v.y, v.z);
        const t = Math.min(1, v.y / 2.2);
        if (t < 0.5) c.copy(lo).lerp(mid, t * 2);
        else c.copy(mid).lerp(hi, (t - 0.5) * 2);
        col.setXYZ(n, c.r, c.g, c.b);
      }
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    this.geo.computeVertexNormals();

    // Ball: the engine's loss at the current weights.
    const here = store.loss;
    const p = this.toScene(a0, b0, here, new Vector3());
    p.y += 0.11;
    this.ball.position.copy(p);

    // Gradient (uphill) and the step lr would take along these two axes.
    const g = store.grads;
    const ga = getParam(g, refA);
    const gb = getParam(g, refB);
    const gLen = Math.hypot(ga, gb);
    if (gLen > 1e-9) {
      const reach = 0.35 + 0.75 * Math.min(1, gLen / 2);
      const tip = p.clone().add(new Vector3((ga / gLen) * reach, 0, (-gb / gLen) * reach));
      this.gradArrow.set(p, tip);
    } else {
      this.gradArrow.group.visible = false;
    }
    const a1 = a0 - s.lr * ga;
    const b1 = b0 - s.lr * gb;
    setParam(work, refA, a1);
    setParam(work, refB, b1);
    const landed = loss(work);
    const inside = Math.abs(a1 - this.centre[0]) <= RANGE && Math.abs(b1 - this.centre[1]) <= RANGE;
    const q = this.toScene(
      Math.max(this.centre[0] - RANGE, Math.min(this.centre[0] + RANGE, a1)),
      Math.max(this.centre[1] - RANGE, Math.min(this.centre[1] + RANGE, b1)),
      landed,
      new Vector3(),
    );
    q.y += 0.085;
    this.ghost.position.copy(q);
    this.ghost.visible = gLen > 1e-9;
    this.stepArrow.set(p, q);

    // Trail of where the ball has been (the surface under it keeps changing, since all 25 parameters move).
    if (this.lastParams !== s.params) {
      this.lastParams = s.params;
      this.trail.push(p.clone());
      if (this.trail.length > TRAIL) this.trail.shift();
    }
    this.trailLine.geometry.dispose();
    this.trailLine.geometry = new BufferGeometry().setFromPoints(this.trail.length > 1 ? this.trail : [p, p]);

    const verdict =
      gLen <= 1e-9
        ? 'Flat here: both gradients are 0.'
        : landed > here
          ? `<span class="d">Overshoot:</span> the step lands higher, at <b>${fmt(landed)}</b>${inside ? '' : ' (off the map)'}.`
          : `The step lands at <b>${fmt(landed)}</b>${inside ? '' : ' (off the map)'}.`;
    this.footer = `Loss here <b>${fmt(here)}</b>. ${verdict}`;
  }

  /** Called once per frame by the main loop; draws only when something changed. */
  frame(): void {
    if (!this.dirty) return;
    if (this.canvas.clientWidth === 0) return;
    this.dirty = false;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (this.canvas.width !== Math.round(w * this.renderer.getPixelRatio())) this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.rebuild();
    const r = 6.3;
    this.camera.position.set(r * Math.cos(this.el) * Math.sin(this.az), r * Math.sin(this.el) + 0.4, r * Math.cos(this.el) * Math.cos(this.az));
    this.camera.lookAt(0, 0.7, 0);
    this.renderer.render(this.scene, this.camera);
  }
}

export function allParamOptions(): { ref: ParamRef; label: string }[] {
  return allRefs().map((ref) => ({ ref, label: paramLabel(ref) }));
}
