/**
 * Camera rig: drag = orbit, right-drag or two fingers = pan, wheel or pinch = zoom,
 * WASD = move, Q/E = turn, plus named presets. Written from scratch so mouse, touch and keys share one path.
 */
import { PerspectiveCamera, Vector3 } from 'three';

export type Pose = { target: [number, number, number]; theta: number; phi: number; radius: number };

const clamp = (x: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, x));

export class CameraRig {
  readonly camera: PerspectiveCamera;
  private target = new Vector3();
  private theta = 0;
  private phi = Math.PI / 2;
  private radius = 12;
  private gTarget = new Vector3();
  private gTheta = 0;
  private gPhi = Math.PI / 2;
  private gRadius = 12;
  private keys = new Set<string>();
  private pointers = new Map<number, { x: number; y: number }>();
  private pinch = 0;
  private dragButton = 0;
  /** Total pointer travel of the current press, to tell a click from a drag. */
  travel = 0;
  minRadius = 3;
  maxRadius = 30;
  /** When true, changes are applied at once instead of gliding. */
  instant = false;
  onInput: () => void = () => {};

  constructor(
    private dom: HTMLElement,
    fov = 38,
  ) {
    this.camera = new PerspectiveCamera(fov, 1, 0.1, 200);
    dom.addEventListener('pointerdown', this.onDown);
    dom.addEventListener('pointermove', this.onMove);
    dom.addEventListener('pointerup', this.onUp);
    dom.addEventListener('pointercancel', this.onUp);
    dom.addEventListener('wheel', this.onWheel, { passive: false });
    dom.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  setPose(p: Pose, instant = false): void {
    this.gTarget.set(...p.target);
    // Take the short way round.
    let d = (p.theta - this.gTheta) % (Math.PI * 2);
    if (d > Math.PI) d -= Math.PI * 2;
    if (d < -Math.PI) d += Math.PI * 2;
    this.gTheta += d;
    this.gPhi = p.phi;
    this.gRadius = p.radius;
    if (instant || this.instant) this.snap();
  }

  private snap(): void {
    this.target.copy(this.gTarget);
    this.theta = this.gTheta;
    this.phi = this.gPhi;
    this.radius = this.gRadius;
    this.place();
  }

  keyDown(code: string): void {
    this.keys.add(code);
  }

  keyUp(code: string): void {
    this.keys.delete(code);
  }

  clearKeys(): void {
    this.keys.clear();
  }

  private onDown = (e: PointerEvent): void => {
    this.dom.setPointerCapture(e.pointerId);
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    this.dragButton = e.button;
    if (this.pointers.size === 1) this.travel = 0;
    if (this.pointers.size === 2) this.pinch = this.pinchDistance();
  };

  private onMove = (e: PointerEvent): void => {
    const p = this.pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x;
    const dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    this.travel += Math.abs(dx) + Math.abs(dy);
    if (this.travel < 4) return;
    this.onInput();
    if (this.pointers.size >= 2) {
      const d = this.pinchDistance();
      if (this.pinch > 0) this.gRadius = clamp((this.gRadius * this.pinch) / d, this.minRadius, this.maxRadius);
      this.pinch = d;
      this.pan(dx * 0.5, dy * 0.5);
    } else if (this.dragButton === 2 || e.shiftKey) {
      this.pan(dx, dy);
    } else {
      this.gTheta -= dx * 0.006;
      this.gPhi = clamp(this.gPhi - dy * 0.005, 0.25, Math.PI / 2 + 0.12);
    }
  };

  private onUp = (e: PointerEvent): void => {
    this.pointers.delete(e.pointerId);
    this.pinch = 0;
  };

  private onWheel = (e: WheelEvent): void => {
    // Once the page has scrolled away from the stage, the wheel belongs to the page.
    if (window.scrollY > 4) return;
    e.preventDefault();
    this.onInput();
    this.gRadius = clamp(this.gRadius * Math.exp(e.deltaY * 0.0011), this.minRadius, this.maxRadius);
  };

  private pinchDistance(): number {
    const [a, b] = [...this.pointers.values()];
    return Math.hypot(a.x - b.x, a.y - b.y) || 1;
  }

  private pan(dx: number, dy: number): void {
    const k = (this.gRadius * 0.0016 * this.camera.fov) / 38;
    const right = new Vector3(Math.cos(this.gTheta), 0, -Math.sin(this.gTheta));
    this.gTarget.addScaledVector(right, -dx * k);
    this.gTarget.y = clamp(this.gTarget.y + dy * k, 0.3, 9);
  }

  zoomBy(factor: number): void {
    this.gRadius = clamp(this.gRadius * factor, this.minRadius, this.maxRadius);
  }

  update(dt: number): void {
    if (this.keys.size) {
      const move = this.gRadius * 0.6 * dt;
      const fwd = new Vector3(-Math.sin(this.gTheta), 0, -Math.cos(this.gTheta));
      const right = new Vector3(Math.cos(this.gTheta), 0, -Math.sin(this.gTheta));
      let used = false;
      if (this.keys.has('KeyW')) (this.gTarget.addScaledVector(fwd, move), (used = true));
      if (this.keys.has('KeyS')) (this.gTarget.addScaledVector(fwd, -move), (used = true));
      if (this.keys.has('KeyA')) (this.gTarget.addScaledVector(right, -move), (used = true));
      if (this.keys.has('KeyD')) (this.gTarget.addScaledVector(right, move), (used = true));
      if (this.keys.has('KeyQ')) ((this.gTheta += 1.2 * dt), (used = true));
      if (this.keys.has('KeyE')) ((this.gTheta -= 1.2 * dt), (used = true));
      if (used) this.onInput();
      this.gTarget.x = clamp(this.gTarget.x, -12, 12);
      this.gTarget.z = clamp(this.gTarget.z, -6, 12);
    }
    if (this.instant) return this.snap();
    const k = 1 - Math.exp(-dt * 7);
    this.target.lerp(this.gTarget, k);
    this.theta += (this.gTheta - this.theta) * k;
    this.phi += (this.gPhi - this.phi) * k;
    this.radius += (this.gRadius - this.radius) * k;
    this.place();
  }

  private place(): void {
    const s = Math.sin(this.phi);
    this.camera.position.set(
      this.target.x + this.radius * s * Math.sin(this.theta),
      this.target.y + this.radius * Math.cos(this.phi),
      this.target.z + this.radius * s * Math.cos(this.theta),
    );
    this.camera.lookAt(this.target);
  }
}
