/** Stat chips, the live sentence above the stage, and the stage timeline. */
import { XOR } from '../engine/data';
import { sampleLoss } from '../engine/matrix';
import { TARGET_LOSS } from '../engine/presets';
import { fmt, fmt2 } from '../model/format';
import { LAST_STAGE, phaseOf, STAGE_TITLES, STAGES } from '../model/stepper';
import type { Store } from '../model/store';
import { explodedSay, neuronFacts } from '../scene/exploded';

const list = (xs: number[]): string => xs.map((x) => `<i>${fmt2(x)}</i>`).join(', ');

/** One plain sentence about what the stage is showing, with the live numbers in italics. */
export function describe(store: Store): string {
  const s = store.s;
  const snap = store.snap;

  if (s.view === 'exploded') return explodedSay(neuronFacts(snap, s.exploded.l, s.exploded.j), s.exploded.step);

  if (s.mode === 'train') {
    const t = s.train;
    if (t.diverged) return `The loss blew up after ${t.iter} steps: the learning rate η = ${fmt2(s.lr)} is too high. Lower it and press Reset.`;
    const st = s.seedStatus;
    const seedNote = st.converges
      ? `Seed ${s.seed} is known to reach a mean loss under ${TARGET_LOSS} in ${st.steps} steps at η = ${st.lr}.`
      : `Seed ${s.seed} gets stuck near a mean loss of ${fmt2(st.finalLoss)} at η = ${st.lr}: too many ReLUs are closed for the points that still need fixing. Try another draw.`;
    if (t.iter === 0) return `Train mode: every step averages the gradients of all four points and moves all 25 parameters. ${seedNote}`;
    const solved = store.batch.loss < TARGET_LOSS;
    return `Step <i>${t.iter}</i> of full-batch gradient descent. Mean loss over the four points: <i>${fmt(store.batch.loss)}</i>.${
      solved ? ' All four predictions are close to their targets: XOR is solved.' : ''
    }`;
  }

  const x = `(${snap.x[0]}, ${snap.x[1]})`;
  switch (STAGES[s.stepIndex]) {
    case 'idle':
      return `Pick one of the four XOR points, then press <b>Next</b> to send it through the network one stage at a time. Click any neuron or tube to see its numbers.`;
    case 'input':
      return `The sample is on the table: x = <i>${x}</i>, and the answer we want is y = <i>${snap.y}</i>.`;
    case 'fwd1':
      return `Layer 1: each neuron adds up weight × input plus its bias (z), then ReLU turns anything negative into 0 (a). Activations: ${list(snap.a[1])}.`;
    case 'fwd2':
      return `Layer 2 does the same with layer 1's activations as its inputs. Activations: ${list(snap.a[2])}.`;
    case 'fwd3':
      return `The output neuron has no ReLU, it just adds: ŷ = <i>${fmt(snap.yhat)}</i>. That is the network's answer for ${x}.`;
    case 'loss':
      return `We wanted ${snap.y} and got ${fmt(snap.yhat)}. The gap is <i>${fmt(snap.diff)}</i>; squared, the loss is L = <i>${fmt(snap.loss)}</i>. The square's side is the gap and its area is the loss.`;
    case 'delta3':
      return `Backward starts at the end. If ŷ moved up a little, how fast would L change? δ = 2(ŷ − y) = <i>${fmt(snap.delta[2][0])}</i>.`;
    case 'grad3':
      return `A weight's gradient is the δ of the neuron it feeds times the activation it carried: ∂L/∂w = δ · a. Brighter amber means a larger gradient.`;
    case 'delta2':
      return `Each hidden 2 neuron gets w × δ sent back along its tube, then its ReLU gate lets it through (z > 0) or blocks it. δ: ${list(snap.delta[1])}.`;
    case 'grad2':
      return `Same rule one layer back: ∂L/∂w = δ · a, using layer 2's δ and layer 1's activations.`;
    case 'delta1':
      return `Each hidden 1 neuron feeds three neurons, so it collects three messages and adds them. It can only do that because layer 2's δ already exist: this is why the pass runs backward. δ: ${list(snap.delta[0])}.`;
    case 'grad1':
      return `The first layer's gradients use the inputs themselves: ∂L/∂w = δ · x. An input of 0 gives its weights a gradient of exactly 0.`;
    case 'update': {
      const after = sampleLoss(store.after, s.sample);
      const verdict = after < snap.loss ? 'lower' : after > snap.loss ? 'higher: the step overshot' : 'unchanged';
      return `Every parameter moves by −η × its gradient, with η = ${fmt2(s.lr)}. On this point the loss goes from ${fmt(snap.loss)} to <i>${fmt(after)}</i> (${verdict}). Press Next to start the next iteration.`;
    }
  }
}

export class Readouts {
  private dots: HTMLButtonElement[] = [];

  constructor(
    private store: Store,
    private stats: HTMLElement,
    private say: HTMLElement,
    private timeline: HTMLElement,
  ) {
    stats.innerHTML = `
      <div class="stat hot"><i id="st-l-k">Loss</i><b id="st-l">–</b></div>
      <div class="stat"><i>Output → target</i><b id="st-y">–</b></div>
      <div class="stat"><i id="st-s-k">Stage</i><b id="st-s">–</b></div>`;

    let prevPhase = '';
    for (let i = 1; i <= LAST_STAGE; i++) {
      const phase = phaseOf(i);
      if (prevPhase && phase !== prevPhase) {
        const sep = document.createElement('li');
        sep.className = 'sep';
        sep.setAttribute('aria-hidden', 'true');
        timeline.appendChild(sep);
      }
      prevPhase = phase;
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.type = 'button';
      b.className = phase;
      b.title = STAGE_TITLES[STAGES[i]];
      b.setAttribute('aria-label', `Stage ${i}: ${STAGE_TITLES[STAGES[i]]}`);
      b.addEventListener('click', () => store.goto(i));
      li.appendChild(b);
      timeline.appendChild(li);
      this.dots.push(b);
    }
    store.subscribe(() => this.render());
    this.render();
  }

  render(): void {
    const store = this.store;
    const s = store.s;
    const snap = store.snap;
    const r = store.reveal;
    const $ = (id: string): HTMLElement => this.stats.querySelector<HTMLElement>(`#${id}`)!;

    const train = s.mode === 'train';
    $('st-l-k').textContent = train ? 'Mean loss' : 'Loss, this point';
    $('st-l').textContent = train ? fmt(store.batch.loss) : r.loss ? fmt(snap.loss) : '?';
    $('st-y').innerHTML = `${r.fwd === 3 ? fmt2(snap.yhat) : '?'} <small>→ ${XOR[s.sample].y}</small>`;
    if (s.view === 'exploded') {
      $('st-s-k').textContent = 'Neuron';
      $('st-s').innerHTML = s.exploded.l === 2 ? 'output' : `h${s.exploded.l + 1} <small>#${s.exploded.j + 1}</small>`;
    } else if (train) {
      $('st-s-k').textContent = 'GD steps';
      $('st-s').textContent = String(s.train.iter);
    } else {
      $('st-s-k').textContent = `Iteration ${s.iteration + 1}`;
      $('st-s').innerHTML = `${s.stepIndex} <small>/ ${LAST_STAGE}</small>`;
    }

    const text = describe(store);
    if (this.say.innerHTML !== text) this.say.innerHTML = text;

    this.timeline.hidden = train || s.view === 'exploded';
    this.dots.forEach((b, i) => {
      const idx = i + 1;
      b.classList.toggle('done', idx <= s.stepIndex);
      b.classList.toggle('now', idx === s.stepIndex);
      if (idx === s.stepIndex) b.setAttribute('aria-current', 'step');
      else b.removeAttribute('aria-current');
    });
  }
}
