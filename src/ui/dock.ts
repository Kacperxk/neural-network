/** The dock: descent view, loss curve and decision map. */
import { XOR } from '../engine/data';
import { predict } from '../engine/matrix';
import { allRefs, refKey, sameRef } from '../engine/params';
import { TARGET_LOSS } from '../engine/presets';
import { fmt, fmt2 } from '../model/format';
import type { Store } from '../model/store';
import { DescentView, paramLabel } from '../scene/descent';
import { HEX } from '../scene/theme';

function fit(canvas: HTMLCanvasElement): { g: CanvasRenderingContext2D; w: number; h: number } | null {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const g = canvas.getContext('2d')!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return { g, w, h };
}

const FONT = '10px "Segoe UI", system-ui, sans-serif';

export class Dock {
  readonly descent: DescentView;
  private lossCanvas: HTMLCanvasElement;
  private mapCanvas: HTMLCanvasElement;
  private dirty = true;
  private $: (sel: string) => HTMLElement;

  constructor(
    root: HTMLElement,
    private store: Store,
  ) {
    const options = allRefs()
      .map((r) => `<option value="${refKey(r)}">${paramLabel(r)}</option>`)
      .join('');
    root.innerHTML = `
      <section class="tile descent" id="t-descent" aria-label="Descent view">
        <div class="lab"><span id="d-title">Descent</span><button type="button" id="d-big" title="Enlarge" aria-label="Enlarge the descent view">⤢</button></div>
        <canvas id="d-canvas" aria-label="A loss surface over two chosen parameters, with a ball at the current weights, an amber arrow pointing uphill and a cyan arrow showing where one step would land."></canvas>
        <div class="axes">
          <select id="d-a" aria-label="First parameter (green axis)" style="border-color:${HEX.pos}">${options}</select>
          <select id="d-b" aria-label="Second parameter (violet axis)" style="border-color:${HEX.target}">${options}</select>
        </div>
        <div class="tile-foot" id="d-foot"></div>
      </section>
      <section class="tile" aria-label="Loss curve">
        <div class="lab">Loss curve · log scale</div>
        <canvas id="k-loss" role="img" aria-label="Mean loss against training step"></canvas>
        <div class="tile-foot" id="k-loss-foot"></div>
      </section>
      <section class="tile" aria-label="Decision map">
        <div class="lab">Decision map · <span class="lc">ŷ(x₁, x₂)</span></div>
        <canvas id="k-map" role="img" aria-label="The network's output across the unit square, with the four XOR points"></canvas>
        <div class="tile-foot" id="k-map-foot"></div>
      </section>`;
    this.$ = (sel) => root.querySelector<HTMLElement>(sel)!;
    this.lossCanvas = this.$('#k-loss') as HTMLCanvasElement;
    this.mapCanvas = this.$('#k-map') as HTMLCanvasElement;
    this.descent = new DescentView(this.$('#d-canvas') as HTMLCanvasElement, store);

    const refs = allRefs();
    const pick = (which: 'a' | 'b') => (): void => {
      const sel = this.$(`#d-${which}`) as HTMLSelectElement;
      const ref = refs.find((r) => refKey(r) === sel.value)!;
      const other = which === 'a' ? store.s.descent.b : store.s.descent.a;
      // The two axes must be different parameters.
      if (sameRef(ref, other)) return this.sync();
      store.set({ descent: { ...store.s.descent, [which]: ref } });
    };
    this.$('#d-a').addEventListener('change', pick('a'));
    this.$('#d-b').addEventListener('change', pick('b'));
    this.$('#d-big').addEventListener('click', () => store.set({ descent: { ...store.s.descent, big: !store.s.descent.big } }));

    store.subscribe(() => {
      this.dirty = true;
      this.sync();
    });
    new ResizeObserver(() => {
      this.dirty = true;
    }).observe(root);
    this.sync();
  }

  private sync(): void {
    const d = this.store.s.descent;
    (this.$('#d-a') as HTMLSelectElement).value = refKey(d.a);
    (this.$('#d-b') as HTMLSelectElement).value = refKey(d.b);
    this.$('#t-descent').classList.toggle('big', d.big);
    this.$('#d-big').textContent = d.big ? '⤡' : '⤢';
    this.$('#d-title').textContent = this.store.s.mode === 'train' ? 'Descent · mean loss' : 'Descent · this point';
  }

  frame(): void {
    this.descent.frame();
    const foot = this.$('#d-foot');
    if (foot.innerHTML !== this.descent.footer) foot.innerHTML = this.descent.footer;
    if (!this.dirty) return;
    this.dirty = false;
    this.drawLoss();
    this.drawMap();
  }

  private drawLoss(): void {
    const c = fit(this.lossCanvas);
    if (!c) return;
    const { g, w, h } = c;
    const s = this.store.s;
    const hist = s.train.history;
    const foot = this.$('#k-loss-foot');
    const padL = 30;
    const padB = 14;
    const padT = 6;
    const padR = 6;
    // Log axis from 1e-4 to 10.
    const lo = -4;
    const hi = 1;
    const yOf = (loss: number): number => {
      const e = Math.max(lo, Math.min(hi, Math.log10(Math.max(loss, 1e-12))));
      return padT + ((hi - e) / (hi - lo)) * (h - padT - padB);
    };
    g.font = FONT;
    g.textBaseline = 'middle';
    for (let e = lo; e <= hi; e++) {
      const y = yOf(10 ** e);
      g.strokeStyle = 'rgba(150,170,220,0.12)';
      g.beginPath();
      g.moveTo(padL, y);
      g.lineTo(w - padR, y);
      g.stroke();
      if (e % 2 === 0 || e === hi) {
        g.fillStyle = '#7d88a5';
        g.textAlign = 'right';
        g.fillText(e === 0 ? '1' : e === 1 ? '10' : `1e${e}`.replace('-', '−'), padL - 4, y);
      }
    }
    // Target line.
    g.strokeStyle = HEX.pos;
    g.setLineDash([3, 3]);
    g.beginPath();
    g.moveTo(padL, yOf(TARGET_LOSS));
    g.lineTo(w - padR, yOf(TARGET_LOSS));
    g.stroke();
    g.setLineDash([]);

    if (hist.length < 2) {
      g.fillStyle = '#7d88a5';
      g.textAlign = 'center';
      g.fillText(s.mode === 'train' ? 'Press Run to start descending' : 'Train mode draws this curve', padL + (w - padL - padR) / 2, h / 2 - 8);
      foot.innerHTML = `Dashed line: the target, mean loss ${TARGET_LOSS}.`;
      return;
    }
    const n = hist.length - 1;
    const span = Math.max(50, n);
    const xOf = (i: number): number => padL + (i / span) * (w - padL - padR);
    g.strokeStyle = HEX.target;
    g.lineWidth = 1.6;
    g.beginPath();
    // Draw at most one point per pixel column.
    const stride = Math.max(1, Math.floor(hist.length / (w * 2)));
    for (let i = 0; i < hist.length; i += stride) {
      const x = xOf(i);
      const y = yOf(hist[i]);
      if (i === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    }
    g.lineTo(xOf(n), yOf(hist[n]));
    g.stroke();
    g.lineWidth = 1;
    g.fillStyle = '#fff';
    g.beginPath();
    g.arc(xOf(n), yOf(hist[n]), 2.6, 0, Math.PI * 2);
    g.fill();
    g.fillStyle = '#7d88a5';
    g.textAlign = 'right';
    g.textBaseline = 'alphabetic';
    g.fillText(`step ${span}`, w - padR, h - 2);
    foot.innerHTML = `Step <b>${n}</b>, mean loss <b>${fmt(hist[n])}</b>${hist[n] < TARGET_LOSS ? ' · below target' : ''}`;
  }

  private drawMap(): void {
    const c = fit(this.mapCanvas);
    if (!c) return;
    const { g, w, h } = c;
    const s = this.store.s;
    const size = Math.min(w - 40, h - 16);
    const x0 = (w - size) / 2 + 6;
    const y0 = 4;
    const G = 36;
    const cell = size / G;
    // ŷ = 0 is dark navy, ŷ = 1 is the accent colour; values outside [0, 1] are clamped for colour only.
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        const x1 = (i + 0.5) / G;
        const x2 = 1 - (j + 0.5) / G;
        const t = Math.max(0, Math.min(1, predict(s.params, [x1, x2])));
        const r = Math.round(16 + (95 - 16) * t);
        const gg = Math.round(24 + (212 - 24) * t);
        const b = Math.round(48 + (255 - 48) * t);
        g.fillStyle = `rgb(${r},${gg},${b})`;
        g.fillRect(x0 + i * cell, y0 + j * cell, cell + 0.6, cell + 0.6);
      }
    }
    g.strokeStyle = 'rgba(150,170,220,0.35)';
    g.strokeRect(x0 + 0.5, y0 + 0.5, size - 1, size - 1);

    // The four data points sit slightly inside the corners so they stay visible.
    const inset = 0.09;
    XOR.forEach((p, i) => {
      const px = x0 + (inset + p.x[0] * (1 - 2 * inset)) * size;
      const py = y0 + (1 - (inset + p.x[1] * (1 - 2 * inset))) * size;
      g.beginPath();
      g.arc(px, py, i === s.sample ? 5.5 : 4.5, 0, Math.PI * 2);
      g.fillStyle = p.y === 1 ? '#ffffff' : '#0b0f1a';
      g.fill();
      g.lineWidth = i === s.sample ? 2.2 : 1.3;
      g.strokeStyle = i === s.sample ? HEX.back : '#ffffff';
      g.stroke();
    });
    g.lineWidth = 1;
    g.font = FONT;
    g.fillStyle = '#7d88a5';
    g.textAlign = 'center';
    g.textBaseline = 'alphabetic';
    g.fillText('x₁ →', x0 + size / 2, y0 + size + 11);
    g.save();
    g.translate(x0 - 6, y0 + size / 2);
    g.rotate(-Math.PI / 2);
    g.fillText('x₂ →', 0, 0);
    g.restore();

    const preds = XOR.map((p) => predict(s.params, p.x));
    this.$('#k-map-foot').innerHTML = `White dot wants 1, dark dot wants 0. ŷ: ${preds.map((v) => `<b>${fmt2(v)}</b>`).join(' ')}`;
  }
}
