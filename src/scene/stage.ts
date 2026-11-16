/**
 * The main stage: renderer, loop, picking, and the translation from app state to what is drawn.
 * Animation is stateless: everything is a function of (state, seconds since the stage was entered),
 * so Back, scrubbing and reduced motion need no special cases.
 */
import {
  ACESFilmicToneMapping,
  Color,
  Fog,
  HalfFloatType,
  Mesh,
  MeshBasicMaterial,
  Raycaster,
  Scene,
  SphereGeometry,
  SRGBColorSpace,
  Vector2,
  Vector3,
  WebGLRenderer,
  WebGLRenderTarget,
} from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { allRefs, getParam, N_LAYERS, SIZES } from '../engine/params';
import { fmt2 } from '../model/format';
import type { Selection, Store } from '../model/store';
import { CameraRig, type Pose } from './camera';
import { ExplodedView, neuronFacts } from './exploded';
import { Labels } from './labels';
import { NetworkView, sameSel, type NetFrame } from './network';
import { buildRoom, type Room } from './room';
import { BENCH_TOP, COL_X, COLORS, ease, GAUGE_X, gaugeY, NET_Y, NEURON_R, neuronPos } from './theme';

// Durations in seconds.
const T_FWD = 0.95;
const T_LOSS = 0.8;
const T_DELTA_OUT = 0.45;
const T_BACK_TRAVEL = 0.75;
const T_BACK_STAGGER = 0.6;
const T_GRAD = 0.55;
const T_UPDATE = 1.1;
const T_TWITCH = 1.3;
const GLOW_FLOOR = 0.05;

/** How long a stage's animation runs; Auto-play waits for this plus a reading pause. */
export function stageDuration(index: number): number {
  if (index >= 2 && index <= 4) return T_FWD;
  if (index === 5) return T_LOSS;
  if (index === 6) return T_DELTA_OUT;
  if (index === 8 || index === 10) return T_BACK_TRAVEL + 2 * T_BACK_STAGGER;
  if (index === 7 || index === 9 || index === 11) return T_GRAD;
  if (index === 12) return T_UPDATE;
  return 0.2;
}

const LAB_POSES: Pose[] = [
  { target: [1.35, 2.85, 0], theta: 0, phi: 1.36, radius: 15.8 },
  { target: [0.2, 3.1, 0], theta: -0.8, phi: 1.3, radius: 13.5 },
  { target: [2.4, 3.1, 0], theta: 0.85, phi: 1.3, radius: 13.5 },
  { target: [1.2, 2.4, 0], theta: 0, phi: 0.8, radius: 15 },
];
const EXPLODED_POSES: Pose[] = [
  { target: [0.9, 3.4, 3.3], theta: 0, phi: 1.5, radius: 13.6 },
  { target: [0.9, 3.4, 3.3], theta: -0.5, phi: 1.4, radius: 13.6 },
  { target: [0.9, 3.4, 3.3], theta: 0.5, phi: 1.4, radius: 13.6 },
];

const COLUMN_CAPTIONS = ['input', 'hidden 1 · ReLU', 'hidden 2 · ReLU', 'output · linear'];
const v3 = new Vector3();

export class Stage {
  readonly renderer: WebGLRenderer;
  readonly rig: CameraRig;
  private scene = new Scene();
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;
  private room: Room;
  private net = new NetworkView();
  private exploded = new ExplodedView();
  private labels: Labels;
  private pulses: { mesh: Mesh; mat: MeshBasicMaterial }[] = [];
  private raycaster = new Raycaster();
  private hover: Selection = null;
  private width = 1;
  private height = 1;
  /** Seconds since the current step-through stage (or exploded sub-step) was entered. */
  private stageT = 0;
  private twitchT = Infinity;
  private dim = 1;
  private poseIndex = 0;
  private last = performance.now();
  private slowFrames = 0;
  private quality = 2;
  private frameCb: (dt: number) => void = () => {};
  private running = false;
  private frameCount = 0;

  constructor(
    private canvas: HTMLCanvasElement,
    labelRoot: HTMLElement,
    private store: Store,
  ) {
    this.renderer = new WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.scene.background = new Color(COLORS.bg);
    this.scene.fog = new Fog(COLORS.bg, 17, 44);
    this.room = buildRoom();
    this.scene.add(this.room.group, this.room.lights, this.net.group, this.exploded.group);

    const pulseGeo = new SphereGeometry(1, 16, 12);
    for (let i = 0; i < 12; i++) {
      const mat = new MeshBasicMaterial({ color: 0xffffff });
      const mesh = new Mesh(pulseGeo, mat);
      mesh.visible = false;
      this.scene.add(mesh);
      this.pulses.push({ mesh, mat });
    }

    this.rig = new CameraRig(canvas);
    this.rig.instant = store.s.reducedMotion;
    this.rig.setPose(LAB_POSES[0], true);
    this.labels = new Labels(labelRoot);

    const target = new WebGLRenderTarget(4, 4, { type: HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(this.renderer, target);
    this.composer.addPass(new RenderPass(this.scene, this.rig.camera));
    this.bloom = new UnrealBloomPass(new Vector2(4, 4), 0.55, 0.5, 0.9);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    canvas.addEventListener('pointermove', (e) => this.onPointerMove(e));
    canvas.addEventListener('pointerleave', () => this.setHover(null));
    canvas.addEventListener('pointerup', (e) => this.onPointerUp(e));

    store.subscribe((s, prev) => {
      if (
        s.stepIndex !== prev.stepIndex ||
        s.iteration !== prev.iteration ||
        s.sample !== prev.sample ||
        s.exploded !== prev.exploded ||
        s.view !== prev.view ||
        s.mode !== prev.mode
      ) {
        this.stageT = 0;
      }
      if (s.verify && s.verify !== prev.verify) this.twitchT = 0;
      if (s.view !== prev.view) {
        this.poseIndex = 0;
        this.rig.setPose(this.poses()[0]);
        this.applyView();
      }
      if (s.reducedMotion !== prev.reducedMotion) this.rig.instant = s.reducedMotion;
    });
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  onFrame(cb: (dt: number) => void): void {
    this.frameCb = cb;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.last = performance.now();
    const tick = (now: number): void => {
      const dt = Math.min(0.1, (now - this.last) / 1000);
      this.last = now;
      this.frame(dt, now);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  /** True while the current stage's animation is still running (Auto-play waits for this). */
  get animating(): boolean {
    if (this.store.s.reducedMotion) return false;
    const s = this.store.s;
    const dur = s.view === 'exploded' ? 0.7 : stageDuration(s.stepIndex);
    return this.stageT < dur;
  }

  private poses(): Pose[] {
    return this.store.s.view === 'exploded' ? EXPLODED_POSES : LAB_POSES;
  }

  /** How much further back the camera must sit than on a 16:9 screen. */
  private fit(): number {
    const aspect = this.width / this.height;
    // On a narrow screen the panels cover the lower half, so only the upper part of the frame is free.
    return Math.max(1, (this.width < 1000 ? 1.25 : 1.7) / aspect);
  }

  private fitPose(p: Pose): Pose {
    // Narrow windows need the camera further back to keep the whole network in frame.
    return { ...p, radius: p.radius * this.fit() };
  }

  resetCamera(): void {
    this.poseIndex = 0;
    this.rig.setPose(this.fitPose(this.poses()[0]));
  }

  nextCamera(): number {
    this.poseIndex = (this.poseIndex + 1) % this.poses().length;
    this.rig.setPose(this.fitPose(this.poses()[this.poseIndex]));
    return this.poseIndex;
  }

  private applyView(): void {
    const studio = this.store.s.view === 'studio';
    this.room.group.visible = !studio;
    this.scene.fog = studio ? null : new Fog(COLORS.bg, 17 * this.fit(), 44 * this.fit());
    (this.scene.background as Color).setHex(studio ? 0x101624 : COLORS.bg);
    this.bloom.strength = studio ? 0.4 : 0.62;
  }

  private resize(): void {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    this.width = w;
    this.height = h;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.labels.resize(w, h);
    const cam = this.rig.camera;
    cam.aspect = w / h;
    // The console sits on the right and the dock at the bottom, so the scene is framed a little up and left.
    if (w >= 1000) cam.setViewOffset(w, h, w * 0.14, h * (h < 820 ? 0.04 : 0.0), w, h);
    else cam.setViewOffset(w, h, 0, h * 0.2, w, h);
    cam.updateProjectionMatrix();
    this.rig.maxRadius = 30 * this.fit();
    this.rig.setPose(this.fitPose(this.poses()[this.poseIndex]), true);
    this.applyView();
  }

  // ---- picking -----------------------------------------------------------------------------------

  private pick(e: PointerEvent): Selection {
    const rect = this.canvas.getBoundingClientRect();
    const p = new Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(p, this.rig.camera);
    const hits = this.raycaster.intersectObjects(this.net.pickables, false);
    if (!hits.length) return null;
    // Neurons and biases win over the tubes that end inside them.
    const solid = hits.find((h) => (h.object.userData.sel as Selection)?.kind !== 'weight' && h.distance < hits[0].distance + 0.5);
    return ((solid ?? hits[0]).object.userData.sel as Selection) ?? null;
  }

  private setHover(sel: Selection): void {
    if (sameSel(sel, this.hover)) return;
    this.hover = sel;
    this.canvas.style.cursor = sel ? 'pointer' : '';
  }

  /** Highlight an object from the keyboard, the same way the mouse does on hover. */
  focusObject(sel: Selection): void {
    this.setHover(sel);
  }

  private onPointerMove(e: PointerEvent): void {
    if (e.buttons) return;
    this.setHover(this.pick(e));
  }

  private onPointerUp(e: PointerEvent): void {
    if (this.rig.travel > 5 || e.button !== 0) return;
    const sel = this.pick(e);
    if (this.store.s.view === 'exploded') {
      // In the Exploded view a click on another neuron opens that one.
      if (sel && sel.kind === 'neuron' && sel.col >= 1) this.store.explode(sel.col - 1, sel.i);
      return;
    }
    this.store.select(sel && sameSel(sel, this.store.s.selection) ? null : sel);
  }

  // ---- per-frame -----------------------------------------------------------------------------------

  private frame(dt: number, now: number): void {
    this.frameCb(dt);
    this.stageT += dt;
    this.twitchT += dt;
    this.rig.update(dt);
    this.rig.camera.updateMatrixWorld();

    const s = this.store.s;
    const explodedOn = s.view === 'exploded';
    const dimGoal = explodedOn ? 0.1 : 1;
    this.dim = s.reducedMotion ? dimGoal : this.dim + (dimGoal - this.dim) * (1 - Math.exp(-dt * 8));
    this.net.group.position.z = -3.2 * (1 - (this.dim - 0.1) / 0.9);

    const frame = this.buildFrame();
    this.net.apply(frame, this.rig.camera.position);
    this.drawPulses();

    this.exploded.group.visible = explodedOn;
    if (this.frameCount++ % 30 === 0) {
      this.labels.obstacles = ['dock', 'console', 'inspector']
        .map((id) => document.getElementById(id))
        .filter((el): el is HTMLElement => !!el && !el.hidden && el.offsetWidth > 0)
        .map((el) => el.getBoundingClientRect());
    }
    this.labels.begin();
    if (explodedOn) {
      const f = neuronFacts(this.store.snap, s.exploded.l, s.exploded.j);
      this.exploded.apply(f, s.exploded.step, s.reducedMotion ? Infinity : this.stageT);
      this.exploded.group.updateMatrixWorld();
      this.exploded.labels(this.labels, this.rig.camera, f, s.exploded.step);
    } else {
      this.drawLabels(frame);
    }
    this.labels.end();

    this.composer.render();

    // Adaptive quality: drop pixel ratio, then bloom, if frames stay slow.
    const ms = dt * 1000;
    this.slowFrames = ms > 24 ? this.slowFrames + 1 : Math.max(0, this.slowFrames - 2);
    if (this.slowFrames > 90 && this.quality > 0 && now > 4000) {
      this.quality--;
      this.slowFrames = 0;
      if (this.quality === 1) {
        this.renderer.setPixelRatio(1);
        this.composer.setPixelRatio(1);
      } else {
        this.bloom.enabled = false;
      }
    }
  }

  private time(): number {
    return this.store.s.reducedMotion ? Infinity : this.stageT;
  }

  /** Index of the step-through stage being animated, or -1 when the stepper is not driving the stage. */
  private animIndex(): number {
    const s = this.store.s;
    return s.mode === 'step' && s.view !== 'exploded' ? s.stepIndex : -1;
  }

  private buildFrame(): NetFrame {
    const store = this.store;
    const s = store.s;
    const snap = store.snap;
    const r = store.reveal;
    const t = this.time();
    const idx = this.animIndex();

    // Activations: a column lights up when the forward pulses reach it.
    const act: (number | null)[][] = [];
    for (let col = 0; col < SIZES.length; col++) {
      let known = col === 0 ? r.input : r.fwd >= col;
      if (col >= 1 && idx === col + 1 && t < T_FWD) known = false;
      act.push(snap.a[col].map((a) => (known ? a : null)));
    }

    // Deltas: output first, then each neuron as its backward pulses arrive.
    const delta: (number | null)[][] = [];
    for (let l = 0; l < N_LAYERS; l++) {
      delta.push(
        snap.delta[l].map((d, j) => {
          if (l < r.deltaFrom) return null;
          if (idx === 6 && l === 2 && t < T_DELTA_OUT * 0.6) return null;
          if ((idx === 8 && l === 1) || (idx === 10 && l === 0)) {
            if (t < j * T_BACK_STAGGER + T_BACK_TRAVEL) return null;
          }
          return d;
        }),
      );
    }

    // Gradient glow, normalised by the largest gradient magnitude in this pass.
    const grads = store.grads;
    let maxG = 0;
    for (const ref of allRefs()) maxG = Math.max(maxG, Math.abs(getParam(grads, ref)));
    const gradGlow: number[][][] = [];
    const biasGlow: number[][] = [];
    for (let l = 0; l < N_LAYERS; l++) {
      let k = l >= r.gradFrom ? 1 : 0;
      if (idx === 11 - 2 * l) k = ease(t / T_GRAD);
      // Relative to the largest gradient of the pass, but never brighter than its real size warrants:
      // once every gradient is below GLOW_FLOOR the whole network dims, as it should near the minimum.
      const norm = (g: number): number => (maxG > 0 && k > 0 ? (k * Math.abs(g)) / Math.max(maxG, GLOW_FLOOR) : 0);
      gradGlow.push(grads.W[l].map((row) => row.map(norm)));
      biasGlow.push(grads.b[l].map(norm));
    }

    // Weights: at the update stage they glide to their new values.
    let w = s.params.W;
    let b = s.params.b;
    if (r.updated) {
      const after = store.after;
      const u = ease(t / T_UPDATE);
      w = w.map((layer, l) => layer.map((row, j) => row.map((x, k) => x + (after.W[l][j][k] - x) * u)));
      b = b.map((layer, l) => layer.map((x, j) => x + (after.b[l][j] - x) * u));
    }

    let twitch: NetFrame['twitch'] = null;
    if (s.verify && this.twitchT < T_TWITCH && !s.reducedMotion) {
      const ref = s.verify.ref;
      const sel: Selection = ref.kind === 'w' ? { kind: 'weight', l: ref.l, j: ref.j, k: ref.k } : { kind: 'bias', l: ref.l, j: ref.j };
      const tau = this.twitchT / T_TWITCH;
      // The real nudge is 1e-5, far too small to see; the wobble is exaggerated on purpose.
      twitch = { sel, amount: Math.sin(tau * Math.PI * 4) * 0.5 * (1 - tau) };
    }

    return {
      w,
      b,
      act,
      delta,
      gradGlow,
      biasGlow,
      dim: this.dim,
      selection: s.selection,
      hover: this.hover,
      yhat: act[3][0],
      target: r.input ? snap.y : null,
      gap: r.loss ? (idx === 5 ? ease(t / T_LOSS) : 1) : 0,
      twitch,
    };
  }

  private drawPulses(): void {
    const s = this.store.s;
    const snap = this.store.snap;
    const idx = this.animIndex();
    const t = this.time();
    let n = 0;
    const emit = (pos: Vector3, color: number, size: number, level: number): void => {
      const p = this.pulses[n++];
      if (!p) return;
      p.mesh.visible = true;
      p.mesh.position.copy(pos);
      p.mesh.scale.setScalar(size);
      p.mat.color.setHex(color).multiplyScalar(level);
    };

    if (idx >= 2 && idx <= 4 && t < T_FWD) {
      // Forward: each pulse carries w * a from the previous column.
      const l = idx - 2;
      const u = ease(t / T_FWD);
      for (let j = 0; j < SIZES[l + 1]; j++) {
        for (let k = 0; k < SIZES[l]; k++) {
          const p = snap.prod[l][j][k];
          const size = p === 0 ? 0.05 : 0.07 + 0.09 * Math.min(1, Math.sqrt(Math.abs(p)));
          emit(this.net.along(l, j, k, u, v3), p === 0 ? 0x5a6788 : COLORS.fwd, size, p === 0 ? 0.8 : 1.7);
        }
      }
    } else if (idx === 6 && t < T_DELTA_OUT) {
      // The loss hands its slope back to the output neuron.
      const u = ease(t / T_DELTA_OUT);
      const out = neuronPos(3, 0);
      v3.set(GAUGE_X, gaugeY(snap.yhat), 0).lerp(new Vector3(out.x + NEURON_R, out.y, 0), u);
      emit(v3, COLORS.back, 0.12, 1.8);
    } else if (idx === 8 || idx === 10) {
      // Backward: neuron k of the earlier column collects w * delta from every neuron it feeds.
      const l = idx === 8 ? 2 : 1;
      let maxTerm = 0;
      for (const terms of snap.aTerms[l]) for (const term of terms) maxTerm = Math.max(maxTerm, Math.abs(term.amount));
      for (let k = 0; k < SIZES[l]; k++) {
        const tk = t - k * T_BACK_STAGGER;
        if (tk < 0 || tk > T_BACK_TRAVEL) continue;
        const u = 1 - ease(tk / T_BACK_TRAVEL);
        for (const term of snap.aTerms[l][k]) {
          const m = maxTerm > 0 ? Math.abs(term.amount) / maxTerm : 0;
          emit(this.net.along(l, term.j, k, u, v3), m === 0 ? 0x5a6788 : COLORS.back, 0.06 + 0.09 * Math.sqrt(m), m === 0 ? 0.8 : 0.7 + 1.3 * m);
        }
      }
    }
    for (; n < this.pulses.length; n++) this.pulses[n].mesh.visible = false;
    if (s.view === 'exploded') for (const p of this.pulses) p.mesh.visible = false;
  }

  private drawLabels(f: NetFrame): void {
    const store = this.store;
    const s = store.s;
    const snap = store.snap;
    const r = store.reveal;
    const cam = this.rig.camera;
    const L = this.labels;
    if (this.dim < 0.9) return;

    for (let col = 0; col < SIZES.length; col++) {
      // Column captions need room; on a phone the legend covers them instead.
      if (this.width >= 700) L.put(`cap${col}`, v3.set(COL_X[col], BENCH_TOP + 0.02, 0.5), COLUMN_CAPTIONS[col], cam, 'cap', 't');
      for (let i = 0; i < SIZES[col]; i++) {
        const pos = neuronPos(col, i);
        const a = f.act[col][i];
        let html = '';
        if (col === 0) {
          html = a === null ? `x<sub>${i + 1}</sub>` : `<span class="a">x<sub>${i + 1}</sub> = ${a}</span>`;
        } else if (a !== null) {
          const z = snap.z[col - 1][i];
          html = col === 3 ? `<span class="a">ŷ = ${fmt2(a)}</span>` : `<span class="z">z ${fmt2(z)}</span> <span class="a">a ${fmt2(a)}</span>`;
        }
        const d = col >= 1 ? f.delta[col - 1][i] : null;
        if (d !== null) html += `${html ? '<br>' : ''}<span class="d">δ ${fmt2(d)}</span>`;
        if (html) L.put(`n${col}_${i}`, v3.copy(pos).setY(pos.y - NEURON_R - 0.08), html, cam, 'node', 't');
      }
    }

    // Loss gauge.
    if (f.target !== null) L.put('target', v3.set(GAUGE_X - 0.3, gaugeY(snap.y), 0), `<span class="y">y = ${snap.y}</span>`, cam, 'node', 'r');
    if (r.loss && f.gap > 0.6) {
      const side = Math.abs(gaugeY(snap.yhat) - gaugeY(snap.y));
      const c = this.net.squareCentre(snap.yhat, snap.y);
      L.put(
        'loss',
        v3.set(c.x + side / 2 + 0.15, c.y, 0),
        `<span class="y">gap ${fmt2(snap.diff)}<br><b>L = ${fmt2(snap.loss)}</b></span>`,
        cam,
        'node',
        'l',
      );
    }

    // Weight and bias labels: the selected one, the hovered one, or all of them when the toggle is on.
    const after = store.after;
    for (let l = 0; l < N_LAYERS; l++) {
      for (let j = 0; j < SIZES[l + 1]; j++) {
        for (let k = 0; k < SIZES[l]; k++) {
          const sel: Selection = { kind: 'weight', l, j, k };
          const focus = sameSel(sel, s.selection) || sameSel(sel, this.hover);
          if (!focus && !s.showWeightLabels) continue;
          const w = s.params.W[l][j][k];
          let html = `<span class="${w >= 0 ? 'w' : 'neg'}">${focus ? 'w = ' : ''}${fmt2(w)}</span>`;
          if (focus && r.updated) html += ` → <span class="${after.W[l][j][k] >= 0 ? 'w' : 'neg'}">${fmt2(after.W[l][j][k])}</span>`;
          if (focus && l >= r.gradFrom) html += `<br><span class="d">∂L/∂w ${fmt2(store.grads.W[l][j][k])}</span>`;
          // Labels sit a third of the way along so the three tubes into one neuron do not stack their labels.
          L.put(`w${l}_${j}_${k}`, this.net.along(l, j, k, focus ? 0.5 : 0.3, v3), html, cam, focus ? 'node focus' : 'tiny', 'c');
        }
        const bsel: Selection = { kind: 'bias', l, j };
        const bfocus = sameSel(bsel, s.selection) || sameSel(bsel, this.hover);
        if (bfocus || s.showWeightLabels) {
          const b = s.params.b[l][j];
          let html = `<span class="${b >= 0 ? 'w' : 'neg'}">${bfocus ? 'b = ' : ''}${fmt2(b)}</span>`;
          if (bfocus && r.updated) html += ` → ${fmt2(after.b[l][j])}`;
          if (bfocus && l >= r.gradFrom) html += `<br><span class="d">∂L/∂b ${fmt2(store.grads.b[l][j])}</span>`;
          const p = this.net.anchorOf(bsel)!;
          L.put(`b${l}_${j}`, v3.copy(p).setY(p.y + 0.2), html, cam, bfocus ? 'node focus' : 'tiny', 'b');
        }
      }
    }
  }
}

export { NET_Y };
