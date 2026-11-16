import 'katex/dist/katex.min.css';
import './styles.css';
import { Article } from './article/article';
import { Store, SPEEDS } from './model/store';
import { Stage, stageDuration } from './scene/stage';
import { Console } from './ui/console';
import { Dock } from './ui/dock';
import { Inspector } from './ui/inspector';
import { Legend, ObjectList } from './ui/legend';
import { Readouts } from './ui/readouts';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
const store = new Store(motionQuery.matches);
motionQuery.addEventListener('change', () => store.set({ reducedMotion: motionQuery.matches, autoplay: false }));

const stage = new Stage($<HTMLCanvasElement>('gl'), $('labels'), store);

const toggleLegend = (): void => store.set({ legendOpen: !store.s.legendOpen });
new Console($('console'), store, {
  camera: () => stage.nextCamera(),
  legend: toggleLegend,
  dock: () => document.body.classList.toggle('dock-open'),
});
new Readouts(store, $('stats'), $('say'), $('timeline'));
new Inspector($('inspector'), store);
const dock = new Dock($('dock'), store);
new Legend($('legend'), store);
new ObjectList($('objects'), store, (sel) => stage.focusObject(sel));

store.subscribe((s, prev) => {
  if (s.view !== prev.view) document.body.classList.toggle('studio', s.view === 'studio');
});

// ---- the clock: auto-play and the training loop ---------------------------------------------------

let autoWait = 0;
let trainDebt = 0;
const article = new Article($('article'), store, (id) => {
  if (id === 'forward') store.goto(4);
  else if (id === 'loss') store.goto(5);
  else if (id === 'delta1') store.goto(10);
  else if (id === 'descent') store.set({ descent: { ...store.s.descent, big: true } });
  else if (id === 'exploded') store.explode(store.s.exploded.l, store.s.exploded.j);
  else if (id === 'train') {
    store.setMode('train');
    if (!store.s.train.running) store.trainToggle();
  } else if (id === 'verify') {
    store.goto(7);
    store.select({ kind: 'weight', l: 2, j: 0, k: 0 });
    store.verify({ kind: 'w', l: 2, j: 0, k: 0 });
  }
  window.scrollTo({ top: 0 });
});

stage.onFrame((dt) => {
  const s = store.s;
  article.frame(dt);
  dock.frame();
  if (s.mode === 'train' && s.train.running) {
    trainDebt += dt * SPEEDS[s.train.speed];
    const n = Math.min(12, Math.floor(trainDebt));
    if (n > 0) {
      trainDebt -= n;
      store.trainStep(n);
    }
  } else {
    trainDebt = 0;
  }
  if (s.autoplay && s.mode === 'step') {
    // Wait for the stage's own animation, then leave time to read the sentence.
    if (stage.animating) autoWait = 0;
    else {
      autoWait += dt;
      const pause = s.view === 'exploded' ? 2.6 : s.reducedMotion ? 2.4 + stageDuration(s.stepIndex) : 1.9;
      if (autoWait > pause) {
        autoWait = 0;
        store.next();
      }
    }
  } else {
    autoWait = 0;
  }
});

// ---- keyboard ---------------------------------------------------------------------------------------

const MOVE_KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE']);

function typingTarget(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  if (!t) return false;
  if (t.tagName === 'SELECT' || t.tagName === 'TEXTAREA') return true;
  return t.tagName === 'INPUT' && (t as HTMLInputElement).type !== 'range';
}

window.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || typingTarget(e)) return;
  // Shortcuts belong to the stage; once the reader is down in the article, leave the keys alone.
  if (window.scrollY > window.innerHeight * 0.6) return;
  const onButton = (e.target as HTMLElement | null)?.tagName === 'BUTTON';
  const onRange = (e.target as HTMLElement | null)?.tagName === 'INPUT';

  if (MOVE_KEYS.has(e.code) && !e.shiftKey) {
    stage.rig.keyDown(e.code);
    e.preventDefault();
    return;
  }
  let handled = true;
  switch (e.key) {
    case 'ArrowRight':
    case 'n':
    case 'N':
      if (onRange && e.key === 'ArrowRight') return;
      store.next();
      break;
    case 'ArrowLeft':
    case 'b':
    case 'B':
      if (onRange && e.key === 'ArrowLeft') return;
      store.back();
      break;
    case ' ':
      if (onButton) return;
      store.toggleAuto();
      break;
    case '1':
    case '2':
    case '3':
    case '4':
      store.setSample(Number(e.key) - 1);
      break;
    case 't':
    case 'T':
      store.setMode(store.s.mode === 'step' ? 'train' : 'step');
      break;
    case 'V':
      store.setView(store.s.view === 'lab' ? 'exploded' : store.s.view === 'exploded' ? 'studio' : 'lab');
      break;
    case 'v': {
      const sel = store.s.selection;
      if (sel && sel.kind === 'weight') store.verify({ kind: 'w', l: sel.l, j: sel.j, k: sel.k });
      else if (sel && sel.kind === 'bias') store.verify({ kind: 'b', l: sel.l, j: sel.j });
      else {
        // Nothing selected yet: pick a last-layer weight so the key always does something visible.
        store.select({ kind: 'weight', l: 2, j: 0, k: 0 });
        store.verify({ kind: 'w', l: 2, j: 0, k: 0 });
      }
      break;
    }
    case 'x':
    case 'X': {
      const sel = store.s.selection;
      if (store.s.view === 'exploded') store.setView('lab');
      else if (sel && sel.kind === 'neuron' && sel.col >= 1) store.explode(sel.col - 1, sel.i);
      else store.setView('exploded');
      break;
    }
    case '[':
      store.setLr(store.s.lr - (store.s.lr > 0.2 ? 0.05 : 0.01));
      break;
    case ']':
      store.setLr(store.s.lr + (store.s.lr >= 0.2 ? 0.05 : 0.01));
      break;
    case 'g':
    case 'G':
      store.randomise();
      break;
    case 'r':
    case 'R':
      stage.resetCamera();
      break;
    case 'c':
    case 'C':
      stage.nextCamera();
      break;
    case '?':
    case '/':
      toggleLegend();
      break;
    case 'Escape':
      if (store.s.legendOpen) toggleLegend();
      else if (store.s.view === 'exploded') store.setView('lab');
      else if (store.s.selection) store.select(null);
      else handled = false;
      break;
    case 'PageDown':
      if (window.scrollY < 4) document.getElementById('article')?.scrollIntoView();
      else handled = false;
      break;
    default:
      handled = false;
  }
  if (handled) e.preventDefault();
});
window.addEventListener('keyup', (e) => stage.rig.keyUp(e.code));
window.addEventListener('blur', () => stage.rig.clearKeys());

$('readon').addEventListener('click', () => document.getElementById('article')?.scrollIntoView());

// Handy for the screenshot script and for poking at the app from the browser console.
(window as unknown as { backprop: unknown }).backprop = { store, stage };

stage.start();
requestAnimationFrame(() => requestAnimationFrame(() => $('loading').classList.add('done')));
