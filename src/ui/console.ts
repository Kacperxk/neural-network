/** The console card: sample chips, mode, view, step controls, learning rate, starting weights. */
import { XOR } from '../engine/data';
import { PRESETS } from '../engine/presets';
import { fmt2 } from '../model/format';
import { LAST_STAGE, STAGE_TITLES, STAGES } from '../model/stepper';
import { explodedSteps, LR_MAX, LR_MIN, SPEEDS, type Store, type View } from '../model/store';

const VIEWS: View[] = ['lab', 'exploded', 'studio'];
const VIEW_NAMES: Record<View, string> = { lab: 'Lab', exploded: 'Exploded', studio: 'Studio' };

function seg(el: HTMLElement, pressed: number): void {
  el.querySelectorAll('button').forEach((b, i) => b.setAttribute('aria-pressed', String(i === pressed)));
}

export type ConsoleHooks = { camera: () => void; legend: () => void; dock: () => void };

export class Console {
  private $: (sel: string) => HTMLElement;

  constructor(
    root: HTMLElement,
    private store: Store,
    hooks: ConsoleHooks,
  ) {
    root.innerHTML = `
      <div class="grp">
        <div class="lab">XOR point <kbd>1 2 3 4</kbd>
          <span class="tools">
            <button type="button" class="mini" id="c-dock" title="Show or hide the charts" hidden>Charts</button>
            <button type="button" class="mini" id="c-cam" title="Cinematic camera (C). R resets the view.">Camera <kbd>C</kbd></button>
            <button type="button" class="mini" id="c-legend" title="How to read the scene (?)">Legend <kbd>?</kbd></button>
          </span>
        </div>
        <div class="seg" id="c-samples" role="group" aria-label="XOR point">
          ${XOR.map((s, i) => `<button type="button" aria-label="Point ${i + 1}: inputs ${s.x[0]} and ${s.x[1]}, target ${s.y}">${s.x[0]},${s.x[1]} → ${s.y}</button>`).join('')}
        </div>
      </div>
      <div class="cols">
        <div class="grp">
          <div class="lab">Mode <kbd>T</kbd></div>
          <div class="seg" id="c-mode" role="group" aria-label="Mode"><button type="button">Step</button><button type="button">Train</button></div>
        </div>
        <div class="grp">
          <div class="lab">View <kbd>⇧V</kbd></div>
          <div class="seg" id="c-view" role="group" aria-label="View">${VIEWS.map((v) => `<button type="button">${VIEW_NAMES[v]}</button>`).join('')}</div>
        </div>
      </div>
      <div class="grp">
        <div class="lab"><span id="c-stage" class="clip">Stage</span> <kbd id="c-stage-keys">← → space</kbd></div>
        <div class="row" id="c-step">
          <button type="button" class="btn" id="c-back">◀ Back</button>
          <button type="button" class="btn primary grow" id="c-next">Next ▶</button>
          <button type="button" class="btn" id="c-auto" aria-pressed="false">Auto-play</button>
        </div>
        <div class="row" id="c-train" hidden>
          <button type="button" class="btn primary grow" id="c-run">Run</button>
          <button type="button" class="btn" id="c-one">+1 step</button>
          <div class="seg" id="c-speed" role="group" aria-label="Training speed">${SPEEDS.map((v) => `<button type="button" title="${v} steps per second">${v}/s</button>`).join('')}</div>
        </div>
      </div>
      <div class="grp">
        <label class="lab" for="c-lr">Learning rate <span class="lc">η</span> <kbd>[ ]</kbd> <output id="c-lr-out"></output></label>
        <input type="range" id="c-lr" min="${LR_MIN}" max="${LR_MAX}" step="0.01" />
      </div>
      <div class="grp">
        <div class="lab">Starting weights <kbd>G</kbd> <span class="val" id="c-seed"></span></div>
        <div class="seg tight" id="c-presets" role="group" aria-label="Starting weights">
          ${PRESETS.map((p) => `<button type="button" title="${p.note} Seed ${p.seed}, ${p.steps} steps to converge.">${p.name}</button>`).join('')}
        </div>
        <div class="row">
          <button type="button" class="btn grow" id="c-random">Randomise</button>
          <button type="button" class="btn grow" id="c-reset">Reset</button>
          <button type="button" class="btn grow" id="c-wl" aria-pressed="false" title="Show every weight's value on its tube">Show values</button>
        </div>
      </div>`;

    this.$ = (sel) => root.querySelector<HTMLElement>(sel)!;
    const on = (sel: string, fn: () => void): void => this.$(sel).addEventListener('click', fn);

    this.$('#c-samples').querySelectorAll('button').forEach((b, i) => b.addEventListener('click', () => store.setSample(i)));
    this.$('#c-mode').querySelectorAll('button').forEach((b, i) => b.addEventListener('click', () => store.setMode(i === 0 ? 'step' : 'train')));
    this.$('#c-view').querySelectorAll('button').forEach((b, i) => b.addEventListener('click', () => store.setView(VIEWS[i])));
    this.$('#c-presets').querySelectorAll('button').forEach((b, i) => b.addEventListener('click', () => store.setPreset(i)));
    this.$('#c-speed').querySelectorAll('button').forEach((b, i) => b.addEventListener('click', () => store.setSpeed(i)));
    on('#c-back', () => store.back());
    on('#c-next', () => store.next());
    on('#c-auto', () => store.toggleAuto());
    on('#c-run', () => store.trainToggle());
    on('#c-one', () => store.trainStep(1));
    on('#c-random', () => store.randomise());
    on('#c-reset', () => store.resetWeights());
    on('#c-wl', () => store.set({ showWeightLabels: !store.s.showWeightLabels }));
    on('#c-cam', hooks.camera);
    on('#c-legend', hooks.legend);
    on('#c-dock', hooks.dock);
    const lr = this.$('#c-lr') as HTMLInputElement;
    lr.addEventListener('input', () => store.setLr(Number(lr.value)));

    const mobile = window.matchMedia('(max-width: 900px)');
    const syncMobile = (): void => {
      this.$('#c-dock').hidden = !mobile.matches;
    };
    mobile.addEventListener('change', syncMobile);
    syncMobile();

    store.subscribe(() => this.render());
    this.render();
  }

  render(): void {
    const s = this.store.s;
    const train = s.mode === 'train';
    const exploded = s.view === 'exploded';
    seg(this.$('#c-samples'), s.sample);
    seg(this.$('#c-mode'), train ? 1 : 0);
    seg(this.$('#c-view'), VIEWS.indexOf(s.view));
    seg(this.$('#c-presets'), PRESETS.findIndex((p) => p.name === s.presetName));
    seg(this.$('#c-speed'), s.train.speed);

    const stepRow = this.$('#c-step');
    const trainRow = this.$('#c-train');
    stepRow.hidden = train && !exploded;
    trainRow.hidden = !train || exploded;
    const back = this.$('#c-back') as HTMLButtonElement;
    const auto = this.$('#c-auto');
    if (exploded) {
      const n = explodedSteps(s.exploded.l) - 1;
      this.$('#c-stage').textContent = `Inside the neuron · step ${s.exploded.step}/${n}`;
      back.disabled = s.exploded.step === 0;
      this.$('#c-next').textContent = s.exploded.step === n ? 'Replay ↺' : 'Next ▶';
    } else if (train) {
      this.$('#c-stage').textContent = `Full-batch descent · step ${s.train.iter}`;
      this.$('#c-run').textContent = s.train.running ? 'Pause' : 'Run';
      (this.$('#c-run') as HTMLButtonElement).disabled = s.train.diverged;
      (this.$('#c-one') as HTMLButtonElement).disabled = s.train.diverged || s.train.running;
    } else {
      this.$('#c-stage').innerHTML = `Stage ${s.stepIndex}/${LAST_STAGE} · ${STAGE_TITLES[STAGES[s.stepIndex]].replace('δ', '<span class="lc">δ</span>')}`;
      back.disabled = s.stepIndex === 0;
      this.$('#c-next').textContent = s.stepIndex === LAST_STAGE ? 'Next iteration ▶' : 'Next ▶';
    }
    this.$('#c-stage-keys').textContent = train && !exploded ? 'space · →' : '← → space';
    auto.setAttribute('aria-pressed', String(s.autoplay));
    auto.textContent = s.autoplay ? 'Pause' : 'Auto-play';

    const lr = this.$('#c-lr') as HTMLInputElement;
    if (Number(lr.value) !== s.lr) lr.value = String(s.lr);
    lr.style.setProperty('--fill', `${((s.lr - LR_MIN) / (LR_MAX - LR_MIN)) * 100}%`);
    this.$('#c-lr-out').textContent = fmt2(s.lr);
    this.$('#c-seed').textContent = `seed ${s.seed}${s.seedStatus.converges ? '' : ' · gets stuck'}`;
    this.$('#c-wl').setAttribute('aria-pressed', String(s.showWeightLabels));
  }
}
