/** "How to read the scene": a short legend, and the hidden button list that makes scene objects keyboard-reachable. */
import { N_LAYERS, SIZES } from '../engine/params';
import { fmt } from '../model/format';
import type { Selection, Store } from '../model/store';
import { HEX } from '../scene/theme';

const sw = (color: string, bar = false): string => `<span class="sw${bar ? ' bar' : ''}" style="background:${color}"></span>`;

export class Legend {
  constructor(
    private root: HTMLElement,
    private store: Store,
  ) {
    root.innerHTML = `
      <button type="button" class="btn icon x" aria-label="Close legend">×</button>
      <h2>How to read the scene</h2>
      <dl>
        <dt>${sw(HEX.fwd)} Sphere</dt><dd>A neuron. It glows brighter the larger its activation <span class="a">a</span>; a neuron outputting 0 stays dark. Columns left to right: inputs, hidden 1, hidden 2, output.</dd>
        <dt>${sw(HEX.pos, true)} ${sw(HEX.neg, true)} Tube</dt><dd>A weight. Thickness is its size |w| (square-root scale, full thickness at |w| ≥ 2). <span class="w">Green</span> is positive, <span class="neg">rose</span> is negative.</dd>
        <dt>${sw(HEX.pos)} Small ball</dt><dd>A bias, attached to its neuron. Size is |b|, colour is its sign.</dd>
        <dt>${sw(HEX.fwd)} Cyan pulse</dt><dd>The forward pass. Each pulse carries weight × activation to the next column; a grey pulse carries 0.</dd>
        <dt>${sw(HEX.back)} Amber pulse, ring, glow</dt><dd>The backward pass. A ring means the neuron's <span class="d">δ</span> is known (a grey ring means δ = 0). The amber glow on a tube is its gradient, brightest for the largest gradient in this pass, so it compares tubes within one pass, not between passes. Once every gradient is below 0.05 the glow fades for all of them.</dd>
        <dt>${sw(HEX.target)} Rail and square</dt><dd>The loss. The ring marks the target <span class="y">y</span>, the dot marks the prediction. The bar between them is the gap, and the square built on it has the loss as its area.</dd>
        <dt>Label chips</dt><dd><span class="z">z</span> is the sum before ReLU, <span class="a">a</span> the output after it, <span class="d">δ</span> the error signal ∂L/∂z. Numbers are rounded to 2 places on the stage and 3 in the cards.</dd>
      </dl>
      <p>Click any sphere, tube or small ball for its card. Drag to orbit, right-drag to pan, scroll to zoom. All keys are listed at the end of the article.</p>`;
    root.querySelector('.x')!.addEventListener('click', () => store.set({ legendOpen: false }));
    store.subscribe((s, prev) => {
      if (s.legendOpen !== prev.legendOpen) this.sync();
    });
    this.sync();
  }

  private sync(): void {
    this.root.hidden = !this.store.s.legendOpen;
    if (this.store.s.legendOpen) this.root.querySelector<HTMLElement>('.x')!.focus();
  }
}

/** A visually hidden button per scene object, so Tab can walk the scene and Enter opens a card. */
export class ObjectList {
  private buttons: { sel: Exclude<Selection, null>; el: HTMLButtonElement }[] = [];

  constructor(
    root: HTMLElement,
    private store: Store,
    onFocus: (sel: Selection) => void,
  ) {
    const add = (sel: Exclude<Selection, null>): void => {
      const el = document.createElement('button');
      el.type = 'button';
      el.addEventListener('click', () => store.select(sel));
      el.addEventListener('focus', () => onFocus(sel));
      el.addEventListener('blur', () => onFocus(null));
      root.appendChild(el);
      this.buttons.push({ sel, el });
    };
    for (let col = 0; col < SIZES.length; col++) for (let i = 0; i < SIZES[col]; i++) add({ kind: 'neuron', col, i });
    for (let l = 0; l < N_LAYERS; l++) {
      for (let j = 0; j < SIZES[l + 1]; j++) {
        for (let k = 0; k < SIZES[l]; k++) add({ kind: 'weight', l, j, k });
        add({ kind: 'bias', l, j });
      }
    }
    store.subscribe((s, prev) => {
      if (s.params !== prev.params || s.sample !== prev.sample || s.stepIndex !== prev.stepIndex) this.label();
    });
    this.label();
  }

  private label(): void {
    const p = this.store.s.params;
    const names = ['Input', 'Hidden 1 neuron', 'Hidden 2 neuron', 'Output neuron'];
    for (const { sel, el } of this.buttons) {
      let text: string;
      if (sel.kind === 'neuron') text = sel.col === 3 ? names[3] : `${names[sel.col]} ${sel.i + 1}`;
      else if (sel.kind === 'weight') {
        const src = sel.l === 0 ? `input ${sel.k + 1}` : `hidden ${sel.l} neuron ${sel.k + 1}`;
        const dst = sel.l === 2 ? 'output' : `hidden ${sel.l + 1} neuron ${sel.j + 1}`;
        text = `Weight from ${src} to ${dst}, value ${fmt(p.W[sel.l][sel.j][sel.k])}`;
      } else {
        const dst = sel.l === 2 ? 'output neuron' : `hidden ${sel.l + 1} neuron ${sel.j + 1}`;
        text = `Bias of ${dst}, value ${fmt(p.b[sel.l][sel.j])}`;
      }
      if (el.textContent !== text) el.textContent = text;
    }
  }
}
