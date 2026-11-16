# Backprop Room

An interactive 3D explainer of how a tiny neural network learns, with backpropagation as the centrepiece.

One network (2 → 3 → 3 → 1, ReLU hidden layers, linear output, 25 parameters), one task (XOR, four points), one loss (squared error), one optimiser (plain gradient descent). You step through a single training iteration stage by stage, open any neuron or weight to see the chain rule with the live numbers substituted, and can check every gradient against a finite-difference nudge. The aim is that you finish able to do a backward pass by hand on this exact example.

It is a static, browser-only app: Vite, TypeScript, Three.js and KaTeX. No backend, no external images, models or fonts; all geometry is generated in code.

## Credit

**Style inspired by the work of Sael, [sael.net](https://sael.net)**, in particular the explainers Earthquake Tower, The Plane of Focus, Inside an AI Data Center and Prism Room: the dark room with one glowing accent, the console of presets and sliders, the live readouts, and the article that explains what you are looking at.

**This is an independent project.** It is not affiliated with or endorsed by Sael. No code, text, assets or layouts were copied; everything here was written from scratch.

The engine mirrors two small reference projects by the author of this repository:

- **microtensor**, a scalar autograd engine (`Value` with `_backward` closures, topological sort, `+=` accumulation): <https://github.com/Kacperxk/kacper-ml-journey/tree/main/phase0/projects/microtensor>
- **numpy_neural_net**, an explicit forward cache with a hand-derived backward pass and a finite-difference `gradient_check`: <https://github.com/Kacperxk/kacper-ml-journey/tree/main/phase0/projects/numpy_neural_net>

Copies of the relevant files are in [`reference/`](reference).

## Run it

Requires Node 20 or newer.

```bash
npm install
```

```bash
npm run dev
```

Other commands:

| Command | What it does |
| --- | --- |
| `npm test` | Unit tests (Vitest) |
| `npm run build` | Type-check and build to `dist/` |
| `npm run preview` | Serve the built `dist/` locally |
| `npm run seeds` | Re-measure how often plain GD solves XOR from a random start, and list converging seeds |
| `npm run shots` | Build, then take headless screenshots at 1280×720 and 1920×1080 into `shots/` and report console errors and overlapping panels (needs `npx playwright install chromium` once) |

## Deploy to GitHub Pages

The build uses relative asset paths (`base: './'` in `vite.config.ts`), so it works from any sub-path.

1. Push the repository to GitHub with `main` as the default branch.
2. In the repository settings, under **Pages**, set **Source** to **GitHub Actions**.
3. Push to `main`. The workflow in [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) runs the tests, builds, and publishes `dist/`.

To host it anywhere else, run `npm run build` and upload the contents of `dist/`.

## How the numbers are kept honest

- `src/engine/value.ts` is the source of truth: a scalar autograd `Value` with `add`, `mul`, `pow`, `relu`, and `sub` / `neg` derived from them, as in microtensor. Every activation, local derivative, δ, gradient and loss shown on the stage, in the cards and in the article is read from a run of this engine.
- `tests/gradcheck.test.ts` compares the engine's gradients with central finite differences (ε = 1e-5, relative error < 1e-6) on all 25 parameters, for 9 seeds on each of the 4 points and on the batch loss.
- `tests/matrix.test.ts` checks a separate hand-written matrix-form backward pass (`src/engine/matrix.ts`) against the engine to 12 decimal places, and checks that the batch gradient equals the mean of the per-point gradients.
- `tests/explain.test.ts` checks that the Intuition, Math and Code layers of every inspector card quote the same numbers with the same rounding, and that nothing is shown before the step-through has reached it.
- `tests/presets.test.ts` checks that every shipped starting point converges in exactly the documented number of steps.

Two bulk displays, the decision map and the loss landscape, are grids of plain-number forward passes rather than engine graphs; a test pins that fast path to the engine bit for bit.

## Layout of the code

```
src/engine/    value.ts (autograd), mlp.ts, trace.ts (snapshots), matrix.ts, gradcheck.ts, presets.ts
src/model/     store.ts (single app state), stepper.ts (13-stage state machine), format.ts
src/scene/     stage.ts, network.ts, exploded.ts, descent.ts, room.ts, camera.ts, labels.ts
src/ui/        console.ts, readouts.ts, inspector.ts, explain.ts, dock.ts, legend.ts, katex.ts
src/article/   article.ts
tests/         unit tests
scripts/       seed-search.ts, shots.mjs
SPEC.md        the design this was built from
```

## Measured, not assumed

With plain full-batch gradient descent and three hidden neurons per layer, ReLU does not solve XOR from every random start. Over 400 random seeds, 3000 steps each (`npm run seeds`):

| Hidden activation | Bias init | η = 0.1 | η = 0.2 |
| --- | --- | --- | --- |
| ReLU | 0 (as microtensor) | 44 % converge | 36 % |
| ReLU | uniform [0, 0.5) (this app) | 80 % | 77 % |
| tanh | either | ~100 % | ~100 % |

The app keeps ReLU, because the "gate" rule is part of what it teaches, and ships four presets verified to converge. **Randomise** draws a fresh seed, checks it, and says so when the draw gets stuck. The article states the 80 % figure.

## Known limitations

- **Fixed scope.** One architecture, one dataset, one loss, plain gradient descent. No momentum, Adam, mini-batches or regularisation.
- **Not every random start trains.** About 1 in 5 random draws gets stuck with dead ReLUs (see above). The presets always work.
- **Biases start in [0, 0.5), not at 0** as in microtensor. With zero biases the point (0, 0) puts every first-layer sum exactly on the ReLU corner, where the finite-difference check is meaningless.
- **Rounding.** The stage shows 2 decimals and the cards 3, so recomputing from displayed numbers can differ in the last digit.
- **Visual encodings are clamped.** Tube thickness follows √|w| and stops growing at |w| = 2. The loss rail spans −1.2 to 2.2. Gradient glow is relative to the largest gradient of the current pass.
- **The Exploded view draws one sum node** where the engine uses a chain of two-input additions.
- **The Descent view is a 2-D slice** of a 25-D surface with a log-compressed height; its step arrow moves only the two chosen parameters.
- **Verify at a ReLU corner.** If a hidden sum is within 1e-4 of zero, the numeric slope and backprop can legitimately disagree; the card warns about it.
- **Back does not cross iterations** in the step-through: you can step back to the start of the current iteration, not into the previous one.
- **Desktop first.** There is a phone layout (one-finger orbit, two-finger pan and pinch zoom, the console as a bottom sheet), but it is cramped, the charts are behind a button, and it has only been checked in an emulated 390×844 viewport, not on real devices.
- **Performance** depends on the GPU. The stage lowers its pixel ratio and then turns off bloom if frames stay slow; it was measured at about 58 fps at 1280×720 on the development laptop, not on a range of machines.
- **Keyboard shortcuts act on the stage only**, and are switched off while you are scrolled down in the article.
- **Screen readers** get labelled controls, a live region for the stage sentence and a button per scene object, but the 3D picture itself is described only by a static label.

## Licences of dependencies

three (MIT), katex (MIT), vite (MIT), typescript (Apache-2.0), vitest (MIT), vite-node (MIT), playwright (Apache-2.0, dev only). KaTeX's fonts ship with the katex package under the SIL Open Font License.
