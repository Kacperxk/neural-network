/** The inspector card: click a neuron, weight or bias to see its numbers as Intuition / Math / Code. */
import { sameRef, type ParamRef } from '../engine/params';
import { fmtSci } from '../model/format';
import type { Store } from '../model/store';
import { explain, type ExplainContext, type Explanation } from './explain';
import { highlight, tex } from './katex';

type Layer = 'intuition' | 'math' | 'code';
const LAYERS: Layer[] = ['intuition', 'math', 'code'];
const LAYER_NAMES: Record<Layer, string> = { intuition: 'Intuition', math: 'Math', code: 'Code' };

export function contextOf(store: Store): ExplainContext {
  const train = store.s.mode === 'train';
  return {
    snap: store.snap,
    reveal: store.reveal,
    lr: store.s.lr,
    after: store.after,
    batch: train ? { perSample: store.batch.samples.map((x) => x.grads), mean: store.batch.grads } : null,
  };
}

function refName(ref: ParamRef): string {
  return ref.kind === 'w' ? 'w' : 'b';
}

export class Inspector {
  private layer: Layer = 'intuition';
  private lastHtml = '';

  constructor(
    private root: HTMLElement,
    private store: Store,
  ) {
    root.addEventListener('click', (e) => {
      const t = (e.target as HTMLElement).closest<HTMLElement>('[data-act]');
      if (!t) return;
      const act = t.dataset.act!;
      if (act === 'close') store.select(null);
      else if (act === 'verify') {
        const ex = this.current();
        if (ex?.param) store.verify(ex.param);
      } else if (act === 'explode') {
        const ex = this.current();
        if (ex?.neuron) store.explode(ex.neuron.l, ex.neuron.j);
      } else if (LAYERS.includes(act as Layer)) {
        this.layer = act as Layer;
        this.render();
      }
    });
    store.subscribe(() => this.render());
    this.render();
  }

  private current(): Explanation | null {
    const sel = this.store.s.selection;
    return sel ? explain(sel, contextOf(this.store)) : null;
  }

  private verifyHtml(ex: Explanation): string {
    const v = this.store.s.verify;
    if (!v || !ex.param || !sameRef(v.ref, ex.param)) return '';
    const name = refName(v.ref);
    const scope = v.scope === 'batch' ? 'the mean loss over all four points' : 'the loss on this point';
    const ok = v.relError < 1e-6;
    const kink = v.minAbsZ < 1e-4;
    return `<div class="verify" aria-live="polite">
      <table>
        <tr><td>L(${name} + ε)</td><td>${v.nudge.lossPlus.toFixed(10)}</td></tr>
        <tr><td>L(${name} − ε)</td><td>${v.nudge.lossMinus.toFixed(10)}</td></tr>
        <tr><td>slope, (L₊ − L₋) / 2ε</td><td>${v.nudge.numeric.toFixed(8)}</td></tr>
        <tr><td class="d">backprop gradient</td><td>${v.analytic.toFixed(8)}</td></tr>
        <tr><td>relative error</td><td class="${ok ? 'ok' : 'bad'}">${fmtSci(v.relError)}</td></tr>
      </table>
      <p>ε = 0.00001, measured on ${scope}. Two extra forward passes for one number; backprop got all 25 in one backward pass.${
        kink ? ' <span class="bad">A hidden z is within 0.0001 of zero, so a ReLU may flip inside the nudge and the two can disagree here.</span>' : ''
      } The wobble on the stage is exaggerated: the real nudge is far too small to see.</p>
    </div>`;
  }

  render(): void {
    const s = this.store.s;
    const ex = s.view === 'exploded' ? null : this.current();
    if (!ex) {
      if (!this.root.hidden) {
        this.root.hidden = true;
        this.lastHtml = '';
      }
      return;
    }
    let body: string;
    if (this.layer === 'intuition') body = ex.intuition;
    else if (this.layer === 'math') body = ex.math.map((m) => `<div class="eq">${tex(m)}</div>`).join('');
    else body = `<pre>${highlight(ex.code)}</pre>`;

    const tabs = LAYERS.map((l) => `<button type="button" data-act="${l}" aria-pressed="${l === this.layer}">${LAYER_NAMES[l]}</button>`).join('');
    const foot =
      (ex.param ? `<button type="button" class="btn grow" data-act="verify" title="Nudge this number by ±ε and compare slopes (V)">Verify with a nudge</button>` : '') +
      (ex.neuron ? `<button type="button" class="btn grow" data-act="explode" title="Open this neuron's computation graph (X)">Open this neuron</button>` : '');

    const html = `
      <div class="insp-head">
        <div><h2>${ex.title}</h2><p>${ex.subtitle}</p></div>
        <button type="button" class="x" data-act="close" aria-label="Close inspector">×</button>
      </div>
      <div class="seg insp-tabs" role="group" aria-label="Explanation layer">${tabs}</div>
      <div class="insp-body">${body}</div>
      ${foot ? `<div class="insp-foot">${foot}</div>` : ''}
      ${this.verifyHtml(ex)}`;
    this.root.hidden = false;
    if (html !== this.lastHtml) {
      const scroll = this.root.scrollTop;
      this.root.innerHTML = html;
      this.lastHtml = html;
      this.root.scrollTop = scroll;
    }
  }
}
