/** Label chips: DOM elements pinned to 3D points. DOM text stays crisp at any zoom. */
import { type Camera, Vector3 } from 'three';

type Entry = { el: HTMLDivElement; html: string; cls: string; used: boolean };

const v = new Vector3();

export class Labels {
  private entries = new Map<string, Entry>();
  private width = 1;
  private height = 1;
  /** Screen rectangles (panels) that caption chips must not be drawn under. */
  obstacles: { left: number; right: number; top: number; bottom: number }[] = [];

  constructor(private root: HTMLElement) {}

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }

  begin(): void {
    for (const e of this.entries.values()) e.used = false;
  }

  /**
   * Show a chip at a world position.
   * `anchor` says which point of the chip sits on the position: 't' = chip hangs below it, 'b' = sits above, etc.
   */
  put(
    id: string,
    world: Vector3,
    html: string,
    camera: Camera,
    cls = '',
    anchor: 't' | 'b' | 'l' | 'r' | 'c' = 't',
    alpha = 1,
  ): void {
    v.copy(world).project(camera);
    if (v.z > 1 || v.z < -1) return;
    const x = (v.x * 0.5 + 0.5) * this.width;
    const y = (-v.y * 0.5 + 0.5) * this.height;
    if (x < -80 || x > this.width + 80 || y < -40 || y > this.height + 40) return;
    if (cls === 'cap') {
      for (const o of this.obstacles) if (x > o.left - 50 && x < o.right + 50 && y > o.top - 22 && y < o.bottom + 8) return;
    }

    let e = this.entries.get(id);
    if (!e) {
      const el = document.createElement('div');
      el.className = 'chip';
      this.root.appendChild(el);
      e = { el, html: '', cls: '', used: false };
      this.entries.set(id, e);
    }
    e.used = true;
    if (e.html !== html) {
      e.el.innerHTML = html;
      e.html = html;
    }
    const full = `chip ${cls} a-${anchor}`;
    if (e.cls !== full) {
      e.el.className = full;
      e.cls = full;
    }
    e.el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    e.el.style.opacity = alpha >= 0.999 ? '' : alpha.toFixed(2);
    e.el.style.display = '';
  }

  end(): void {
    for (const e of this.entries.values()) if (!e.used) e.el.style.display = 'none';
  }
}
