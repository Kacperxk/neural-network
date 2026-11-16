/**
 * Measures how reliably plain full-batch GD solves XOR on the 2-3-3-1 network,
 * for ReLU and for tanh, and lists seeds that converge. Run with: npm run seeds
 */
import { XOR } from '../src/engine/data';
import { batchGrads, forward } from '../src/engine/matrix';
import { stepParams, type Activation, type Params } from '../src/engine/params';
import { initParams } from '../src/engine/rng';
import { minHiddenAbsZ } from '../src/engine/gradcheck';

const N_SEEDS = 400;
const MAX_STEPS = 3000;
const TARGET = 1e-3;

type Run = { steps: number; loss: number; ok: boolean; diverged: boolean };

function train(seed: number, lr: number, act: Activation): Run {
  let p: Params = initParams(seed);
  let loss = Infinity;
  for (let i = 0; i <= MAX_STEPS; i++) {
    const g = batchGrads(p, act);
    loss = g.loss;
    if (!Number.isFinite(loss) || loss > 1e6) return { steps: i, loss, ok: false, diverged: true };
    if (loss < TARGET) return { steps: i, loss, ok: true, diverged: false };
    p = stepParams(p, g.grads, lr);
  }
  return { steps: MAX_STEPS, loss, ok: false, diverged: false };
}

function deadCount(seed: number): number {
  const p = initParams(seed);
  let dead = 0;
  for (let l = 0; l < 2; l++) {
    for (let j = 0; j < 3; j++) {
      if (XOR.every((s) => forward(p, s.x).z[l][j] <= 0)) dead++;
    }
  }
  return dead;
}

for (const act of ['relu', 'tanh'] as Activation[]) {
  for (const lr of [0.1, 0.2]) {
    let ok = 0;
    let diverged = 0;
    const steps: number[] = [];
    for (let seed = 1; seed <= N_SEEDS; seed++) {
      const r = train(seed, lr, act);
      if (r.ok) {
        ok++;
        steps.push(r.steps);
      }
      if (r.diverged) diverged++;
    }
    steps.sort((a, b) => a - b);
    const med = steps.length ? steps[Math.floor(steps.length / 2)] : NaN;
    console.log(
      `${act} lr=${lr}: ${((100 * ok) / N_SEEDS).toFixed(1)}% reach loss<${TARGET} within ${MAX_STEPS} steps` +
        ` (median ${med} steps), ${((100 * diverged) / N_SEEDS).toFixed(1)}% diverge`,
    );
  }
}

// Candidate presets: ReLU seeds that converge at every tested rate, start with no dead neuron,
// start with a visibly wrong loss, and take long enough to be worth watching.
console.log('\nReLU candidates (seed: steps@0.1 / @0.2 / @0.3, initial loss, min |z| at init):');
let shown = 0;
for (let seed = 1; seed <= N_SEEDS && shown < 40; seed++) {
  const runs = [0.1, 0.2, 0.3].map((lr) => train(seed, lr, 'relu'));
  if (!runs.every((r) => r.ok)) continue;
  if (deadCount(seed) > 0) continue;
  const l0 = batchGrads(initParams(seed), 'relu').loss;
  if (l0 < 0.3 || l0 > 3) continue;
  if (runs[0].steps < 120 || runs[0].steps > 600) continue;
  if (minHiddenAbsZ(initParams(seed), 'batch') < 0.02) continue;
  console.log(`  ${seed}: ${runs.map((r) => r.steps).join(' / ')}  L0=${l0.toFixed(3)} minz=${minHiddenAbsZ(initParams(seed), 'batch').toFixed(3)}`);
  shown++;
}
