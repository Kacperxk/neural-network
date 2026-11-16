# Backprop Room: specification

An interactive 3D explainer of one tiny neural network learning XOR, with
backpropagation as the centrepiece. Static, browser-only, deployable to GitHub
Pages. Independent project; the visual style is inspired by the explainers of
Sael (sael.net). No code, text, assets or layout are taken from them.

Priority order: (1) every number shown is correct, (2) the explanation is clear,
(3) visual and interaction craft.

Goal for the reader: finish able to compute a backward pass by hand on this
exact network.

---

## 1. Fixed scope

| Thing | Choice |
| --- | --- |
| Network | 2 → 3 → 3 → 1, 25 parameters (9 + 12 + 4) |
| Hidden activation | ReLU (decision and measurements in §10) |
| Output | linear |
| Data | XOR, 4 points: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0 |
| Loss | per sample L = (ŷ − y)²; training loss = mean over the 4 samples |
| Optimiser | plain full-batch gradient descent, learning-rate slider |
| Out of scope | Adam, momentum, mini-batches, regularisation, other datasets, other shapes |

## 2. What I took from the references and the Sael pages

**microtensor/engine.py**: a `Value` holds `data`, `grad`, `_children`, `_op`
and a `_backward` closure. `backward()` builds a topological order by DFS, sets
the root grad to 1 and runs the closures in reverse. Every closure uses `+=`,
which is what makes a value with several consumers work. `nn.py` builds a
neuron as `val = b; val += w[i] * x[i]` and then `.relu()`; the last layer is
linear. The TypeScript engine mirrors this line for line.

**numpy_neural_net/model.py**: forward returns a cache of every intermediate;
backward is hand-derived from the cache; `gradient_check` nudges each parameter
by ±eps and compares. The matrix-form module and the Verify button mirror this.

One difference to flag: `layers.relu_backward` treats z = 0 as "on" (`x >= 0`)
while `engine.py` treats it as "off" (`data > 0`). The app uses the engine's
rule (derivative 0 at z = 0) everywhere and says so in the article.

**Sael pages (prism-room, earthquake-tower, data-center, plane-of-focus)**,
observed, not copied:

- A full-viewport stage on #0b0f1a; the model sits on a physical "bench" in a
  dim room, lit warm, with one cool glowing accent.
- Top left: a small uppercase kicker line, a large bold title with one phrase
  in the accent colour, then 3 stat chips with live values.
- Top right: one console card. Labelled groups in tiny uppercase, segmented
  preset buttons, sliders with the live value right-aligned, and the keyboard
  shortcut printed next to each group label.
- Top centre: one live sentence describing the current state in plain words.
- Small label chips float next to objects in the scene with their live value.
- View switch Lab / Exploded / Studio, a camera button, a help button.
- Article in short paragraphs: analogy, then mechanism, always with the live
  numbers; a section that says what is simplified; a one-line key reference.

## 3. Tech

- Vite + TypeScript (strict), Three.js, KaTeX. No framework, no backend.
- Vitest for unit tests. Playwright (dev only) for headless screenshots.
- No external images, models or web fonts. All geometry is procedural. System
  font stack for UI; KaTeX ships its own fonts through npm.
- GitHub Pages: `base: './'` in `vite.config.ts` plus
  `.github/workflows/deploy.yml` (build, upload `dist`, deploy-pages).
- Licences: three (MIT), katex (MIT), vite (MIT), vitest (MIT), playwright
  (Apache-2.0). Listed in the README.

## 4. File layout

```
index.html
vite.config.ts  tsconfig.json  package.json
SPEC.md  README.md
reference/                     copies of microtensor + numpy_neural_net
.github/workflows/deploy.yml
scripts/
  shots.mjs                    Playwright: screenshots at 1280x720 and 1920x1080
  seed-search.ts               finds and verifies converging presets
src/
  main.ts                      boot, loading line, wiring
  styles.css
  engine/
    value.ts                   scalar autograd Value (source of truth)
    mlp.ts                     Neuron / Layer / MLP built only from Value ops
    trace.ts                   one forward+backward run -> a plain Snapshot
    matrix.ts                  explicit forward cache + hand-derived backward
    gradcheck.ts               central finite differences
    rng.ts                     seeded PRNG (mulberry32)
    presets.ts                 verified seeds
    data.ts                    the 4 XOR points
    params.ts                  the 25 parameters as plain numbers
  model/
    store.ts                   single app state, training step, verify
    stepper.ts                 pure step-through state machine
    format.ts                  one number formatter used everywhere
  scene/
    stage.ts                   renderer, loop, picking, pulses, quality scaling
    theme.ts                   colours and layout constants
    room.ts                    bench, floor, back wall, lights
    network.ts                 neurons, tubes, bias satellites, loss gauge
    labels.ts                  DOM label chips projected from 3D
    exploded.ts                one neuron as a computation graph
    camera.ts                  orbit / pan / zoom / WASD / QE / presets
    descent.ts                 second small scene: loss surface slice
  ui/
    console.ts                 presets, sliders, chips, toggles
    readouts.ts                title, stat chips, live sentence
    inspector.ts               card with Intuition / Math / Code layers
    explain.ts                 builds the three layers from one Snapshot
    dock.ts                    descent tile, loss curve, decision map
    legend.ts                  "how to read the scene" + keyboard object list
    katex.ts                   render helper + colour macros
  article/
    article.ts                 sections, each bound to a stage state
tests/
  value.test.ts                op-level derivatives, += accumulation
  gradcheck.test.ts            engine vs finite differences, all 25 params
  matrix.test.ts               hand-written matrix backward vs engine
  presets.test.ts              every shipped seed converges
  stepper.test.ts              state machine: next/back/round-trip
  explain.test.ts              Intuition/Math/Code layers quote the same numbers
```

## 5. Engine (source of truth)

### 5.1 `Value`

```ts
class Value {
  data: number; grad = 0;
  readonly children: Value[]; readonly op: string;
  local: number[];              // d(out)/d(child_i), filled when the op runs
  label?: string;               // e.g. "w2[1][0]", "z1[2]"
  add(o) mul(o) pow(k) relu() neg() sub(o)
  backward(): Value[]           // returns the reverse topological order used
}
```

- Ops create the output, store the local derivatives, and a `_backward` that
  does `child.grad += local[i] * out.grad`. Same structure as microtensor; the
  only addition is that the local derivative is kept as a field so the UI can
  show it instead of recomputing it.
- `sub` and `neg` are derived exactly as in microtensor (`a + (b * -1)`).
- `backward()` is the same DFS topological sort, reversed. It returns the
  order so the Exploded view can animate nodes in the true order.

### 5.2 `mlp.ts`

`Neuron.forward(x)` is `val = b; val = val + w[i]*x[i]` then `relu()` for
hidden layers. It also returns the intermediate nodes (each product, each
partial sum, z, a) so nothing has to be rebuilt for display.
Init: weights uniform(−1, 1) from the seeded PRNG (as microtensor); biases
uniform [0, 0.5) (not 0, see §10).

### 5.3 `trace.ts`: the Snapshot

`traceSample(params, sample)` builds the graph for one sample, calls
`backward()` on `L = (ŷ − y)**2`, and copies everything out into a plain
immutable object:

```ts
type Snapshot = {
  x: number[]; y: number;
  z: number[][]; a: number[][];        // per layer, a[0] = x
  yhat: number; loss: number;
  delta: number[][];                   // dL/dz per neuron = z.grad
  dAct: number[][];                    // relu local derivative (1 or 0)
  dW: number[][][]; db: number[][];    // parameter gradients
  contributions: ...                   // per value: each `+=` term, in order
  graph: NodeInfo[]; order: number[];  // full graph + backward order
};
```

`traceBatch(params)` builds one graph for `L = (L0+L1+L2+L3) * 0.25` and also
returns the four per-sample snapshots, so the UI can show that the batch
gradient is the mean of the per-sample gradients.

Rule: **the UI reads numbers only from Snapshots.** δ is `z.grad`, φ′(z) is
the relu node's `local[0]`, a `+=` contribution is `local[i] * out.grad` of a
real consumer node. Nothing is recomputed with a separate formula.

Two bulk fields would be wasteful to draw through the graph: the decision map
(a grid of forward passes) and the loss-surface slice. They use the plain
float `forward()` from `matrix.ts`, which a test pins to the engine's forward
to 1e-12.

### 5.4 `matrix.ts`

Mirrors `TwoLayerNet`: `forward(params, x)` returns ŷ and a cache;
`backward(params, cache, y)` implements exactly the article's equations:

```
δ_out = 2(ŷ − y)
δ(l)  = (W(l+1)ᵀ δ(l+1)) ⊙ φ′(z(l))
dW(l) = δ(l) a(l−1)ᵀ        db(l) = δ(l)
```

### 5.5 Tests

- `gradcheck.test.ts`: for each preset seed and each of the 4 samples, and for
  the batch loss, compare engine gradients with central differences at
  eps = 1e-5; relative error |a − n| / max(|a| + |n|, tiny) < 1e-6 on all 25
  parameters. Inputs that sit within 1e-4 of a ReLU kink are detected and
  reported, not silently skipped.
- `matrix.test.ts`: matrix-form gradients equal engine gradients (< 1e-12),
  and plain `forward()` equals the engine forward.
- `value.test.ts`: each op's local rule; `a*a` at a = 2 gives grad 4 (the
  accumulation check from microtensor's README).
- `presets.test.ts`: every shipped seed reaches mean loss < 1e-3 in exactly the
  documented number of steps at its learning rate, and starts away from
  every ReLU kink.

## 6. App state and the step-through state machine

### 6.1 Store

```ts
type AppState = {
  params: Params;            // 25 numbers, the only mutable model state
  seed: number;
  lr: number;                // 0.01 .. 1.5, default from preset search
  sample: 0|1|2|3;
  mode: 'step' | 'train';
  view: 'lab' | 'exploded' | 'studio';
  step: StepState;
  selection: null | {kind:'neuron', layer, i} | {kind:'weight', layer, j, k}
                  | {kind:'bias', layer, j};
  train: { running: boolean; iter: number; history: number[] };
  descent: { wA: ParamRef; wB: ParamRef; open: boolean };
  reducedMotion: boolean;
};
```

One `dispatch(action)`; scene and UI subscribe and render from state. No
component keeps its own copy of a number.

### 6.2 Step-through

One training iteration on the selected sample. The Snapshot is computed once
when the iteration starts; the stepper only decides **how much of it is
revealed**. That makes Back trivial and keeps numbers consistent.

```
 idle ─Next→ input ─→ fwd1 ─→ fwd2 ─→ fwd3 ─→ loss
                                                 │
   ┌─────────────────────────────────────────────┘
   ▼
 delta3 ─→ grad3 ─→ delta2 ─→ grad2 ─→ delta1 ─→ grad1 ─→ update ─Next→ input (iteration + 1)
```

| # | Stage | What is revealed | What moves on the stage |
| - | --- | --- | --- |
| 0 | `idle` | weights only | nothing |
| 1 | `input` | x₁, x₂, target y | input nodes light up |
| 2 | `fwd1` | z⁽¹⁾, a⁽¹⁾ | forward pulses input → hidden 1 |
| 3 | `fwd2` | z⁽²⁾, a⁽²⁾ | forward pulses hidden 1 → hidden 2 |
| 4 | `fwd3` | ŷ | forward pulses hidden 2 → output |
| 5 | `loss` | L = (ŷ − y)² | the gap between ŷ and y appears |
| 6 | `delta3` | δ_out = 2(ŷ − y) | output node takes the backward colour |
| 7 | `grad3` | ∂L/∂W⁽³⁾, ∂L/∂b⁽³⁾ | last-layer tubes show their gradient |
| 8 | `delta2` | δ⁽²⁾ | backward pulses output → hidden 2, one neuron at a time |
| 9 | `grad2` | ∂L/∂W⁽²⁾, ∂L/∂b⁽²⁾ | |
| 10 | `delta1` | δ⁽¹⁾ | backward pulses hidden 2 → hidden 1; each δ is a visible sum of 3 arrivals |
| 11 | `grad1` | ∂L/∂W⁽¹⁾, ∂L/∂b⁽¹⁾ | |
| 12 | `update` | w ← w − η ∂L/∂w | tubes thicken or thin to the new \|w\| |

- `StepState = { index: 0..12, iteration, snapshot, before: Params, after: Params }`.
- `next(state)`: index + 1; from `update` it commits `after`, takes a new
  Snapshot and goes to `input`.
- `back(state)`: index − 1; leaving `update` restores `before`. Back from
  `input` goes to `idle`. Going back across an iteration boundary is not
  supported (stated in the UI by disabling the button).
- Changing sample, seed or weights resets to `idle`. Changing the learning
  rate keeps the stage and recomputes `after`.
- Auto-play advances on a timer that waits for the stage animation to finish.
- Pulses are pure animation: they start when a stage is entered and never
  gate the numbers. Reduced motion replaces them with a short fade.
- Brightness of a backward pulse and of a gradient-coloured tube is
  `|gradient| / max |gradient| in this snapshot`, with the scale printed in
  the legend so brightness is not mistaken for an absolute value.

### 6.3 Train mode

Full-batch GD: each tick `traceBatch`, then `w ← w − η · mean gradient`.
Controls: Run / Pause, single Step, Reset, speed (steps per frame).
Shown live: loss curve (log y axis), decision map of the unit square
(ŷ as colour, the 4 data points on top), the four ŷ values next to their
targets. The inspector in Train mode shows the four per-sample gradients of
the selected weight and their mean.

## 7. Scenes

### 7.1 Main stage scene graph

```
Scene
├─ Room                       (hidden in Studio)
│   ├─ Floor, BackWall        dark, slightly glossy
│   ├─ Bench                  slab the network stands on, emissive edge strip
│   └─ Lights                 hemisphere + warm key + cool rim
├─ Network
│   ├─ Columns[0..3]          x = input, hidden 1, hidden 2, output
│   │   └─ Neuron             sphere, emissive ∝ activation
│   │       ├─ BiasSatellite  small sphere orbiting; size ∝ |b|, colour = sign
│   │       └─ Halo           ring, shown when δ is revealed
│   ├─ Weights[l][j][k]       tube, radius ∝ |w|, colour = sign
│   └─ TargetNode + LossGap   marker for y, a bar between ŷ and y, length |ŷ − y|
├─ Pulses                     pooled small spheres moving along tubes
├─ ExplodedGraph              (only in Exploded view) nodes ×, Σ, ReLU + wires
└─ FocusRing                  keyboard focus / hover outline
DOM overlay: label chips (a, z, δ, w) projected each frame
```

Layout: columns along x, neurons spread along y, a small z offset per column
so tubes separate when orbiting. Tube radius is clamped to
[0.012, 0.09] with a square-root map; the clamp is stated in the legend.

Colour language (the same in the scene, the KaTeX and the charts):

| Meaning | Colour |
| --- | --- |
| accent, forward values, activation a | cyan #5fd4ff |
| backward signal, δ, gradients | amber #ffb454 |
| positive weight | teal-green #4fe0b0 |
| negative weight | rose #ff6b8a |
| target y, loss | white / violet #b79cff |

### 7.2 View modes

- **Lab**: the room, bench, all labels. Default.
- **Exploded**: the selected neuron (default: hidden 2, neuron 1) moves
  forward and opens into its graph: one × node per incoming weight, one Σ
  node (with the bias entering it), one ReLU gate, and the outgoing branches
  to each consumer. The rest of the network dims. A separate mini-stepper
  walks the backward pass node by node in the engine's real order:
  1. a.grad collects one `+=` term per consumer (shown arriving one at a time)
  2. ReLU gate: multiply by 1 or 0 → δ = z.grad
  3. Σ distributes δ unchanged to every input of the sum
  4. each × swaps: w gets a·δ, the incoming a gets w·δ
  The engine really uses a chain of two-input adds; the single Σ node is a
  drawing simplification and the Code layer shows the real loop.
- **Studio**: no room, flat backdrop, larger labels. For reading and
  screenshots.

### 7.3 Descent view

A second, small Three.js scene in a dock tile (can be enlarged).

- Surface: mean loss over the 4 samples as a function of two chosen
  parameters, the other 23 held at their current values. 41×41 grid, plain
  `forward()`. Height uses a soft log scale; the real loss is in the readout.
- Ball at the current (w_A, w_B, L). Amber arrow = gradient direction
  (uphill), cyan arrow = the actual step −η·∇ restricted to the two axes.
- Two dropdowns choose the parameters. The window is fixed in parameter space
  and recentres only when the ball leaves it, so an overshoot is visible as
  the ball jumping across the valley.
- A "one step here" button and the shared learning-rate slider; with η high
  the ball visibly climbs. The surface is recomputed when the other weights
  change, and the article says plainly that this is a 2-D slice of a 25-D
  surface.

### 7.4 Camera and input

| Input | Action |
| --- | --- |
| drag / one-finger drag | orbit |
| right-drag / two-finger drag | pan |
| scroll / pinch | zoom |
| W A S D, Q E | move, turn |
| R | reset view |
| C | cycle cinematic presets (wide, forward side, backward side, top) |
| click / tap | select neuron, weight or bias |
| Tab, Shift+Tab, Enter, Esc | walk and select scene objects from the keyboard |
| → or N, ← or B, Space | Next, Back, Auto-play |
| 1 2 3 4 | choose XOR sample |
| T | Step mode ↔ Train mode |
| ⇧V | cycle Lab / Exploded / Studio |
| X | explode the selected neuron |
| [ ] | learning rate down / up |
| V | Verify selected weight |
| G | Randomise weights |
| ? | legend |

Scene objects are mirrored by a visually hidden list of buttons so they are
reachable by Tab and announced by screen readers. Custom camera controller
(no OrbitControls) so the keyboard moves share one code path.

Page scroll vs zoom: the stage is the first 100vh. With the pointer over the
canvas at the top of the page, the wheel zooms. A "Read on ↓" button and
PageDown go to the article; once the page has scrolled, the wheel scrolls
the page. Article buttons ("Show me") scroll back up and set the stage.

## 8. Screen layout (checked at 1280×720 and 1920×1080)

```
┌──────────────────────────────────────────────────────────────────────┐
│ kicker                    live sentence (italic)        ┌─ console ─┐│
│ BIG TITLE                                               │ sample    ││
│ [loss] [ŷ vs y] [step]                                  │ step ctl  ││
│                                                         │ lr slider ││
│                    3D stage                             │ init/view ││
│                                                         └───────────┘│
│                                                         ┌ inspector ┐│
│ ┌descent┐┌loss curve┐┌decision map┐                     │ I / M / C ││
│ └───────┘└──────────┘└────────────┘      read on ↓      └───────────┘│
└──────────────────────────────────────────────────────────────────────┘
  article (single column, max 720px) … keys … footer credit
```

Dock tiles collapse to their title bar. Below 900px wide the console becomes
a bottom sheet and the dock a horizontal scroller.

## 9. Inspector, Verify, article

### 9.1 Inspector

Click a neuron, weight or bias. The card has three layers built **from the
same Snapshot by one function** (`explain.ts`), so they cannot disagree:

- **Intuition**: one or two plain sentences with the live numbers, e.g.
  "This weight carries 0.83 forward. The neuron it feeds wants to move by
  −0.41, so the loss changes by 0.83 × −0.41 for each unit of this weight."
- **Math**: KaTeX, symbols coloured as in the scene, numbers substituted:
  ∂L/∂w = δ_j · a_k = (−0.41)(0.83) = −0.34.
- **Code**: the matching microtensor line(s), e.g. the `__mul__` closure
  `self.grad += other.data * out.grad`, annotated with the same numbers.

Values not yet revealed by the stepper are shown as "?" with a hint to press
Next, so the card never runs ahead of the stage. `explain.test.ts` extracts
the numbers from all three layers and asserts they are identical.

### 9.2 Verify

For the selected weight: evaluate L at w + ε and w − ε (ε = 1e-5, plain
forward), show `(L⁺ − L⁻) / 2ε` next to the backprop gradient and the
relative error. The tube twitches both ways while it runs. If z of any
neuron is within 1e-4 of 0 the card warns that the slope is not defined at
the ReLU kink.

### 9.3 Article sections

Each has a "Show me" button that sets the stage state named on the right.

1. What a network is (forward pass) → step `fwd3`, sample (1,0)
2. Loss: how wrong is it → step `loss`
3. Gradient descent: why the negative gradient → descent tile enlarged
4. Backprop on one neuron (chain rule) → Exploded view
5. Backprop through layers: scalar δ recursion first, then matrix form → step `delta1`
6. The update and the training loop; batch gradient = mean of per-sample → Train mode
7. Verifying gradients → Verify on a chosen weight
8. How accurate it is / what is simplified
9. Keys

Notation, defined on first use and identical everywhere:
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾, a⁽ˡ⁾ = φ(z⁽ˡ⁾), a⁽⁰⁾ = x, ŷ = z⁽³⁾, L = (ŷ − y)²,
δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾, η = learning rate. Indices: w⁽ˡ⁾_jk goes **from** neuron k of
layer l−1 **to** neuron j of layer l. Live numbers in the article are italic
and update with the stage.

## 10. Decision: ReLU kept (measured in milestone 1)

Plain full-batch GD, mean loss target < 1e-3 within 3000 steps, 400 random
seeds (`npm run seeds`):

| Hidden activation | Bias init | lr 0.1 | lr 0.2 |
| --- | --- | --- | --- |
| ReLU | 0 (as microtensor) | 44 % converge | 36 % |
| ReLU | uniform [0, 0.5) | 80 % | 77 % |
| tanh | either | ~100 % | ~100 % |

So ReLU on 3 hidden units is **not** reliable from a random draw. The brief
says to switch to tanh in that case, but the brief also asks for the "ReLU is
a gate, 1 or 0" rule and an engine whose ops are add, mul, pow, relu. The two
pull in opposite directions. **Decided: option A, keep ReLU.**

- **A. Keep ReLU (what is built now, my recommendation).** Engine stays a
  mirror of microtensor, the gate rule stays. Four presets are verified by a
  test to converge (133 to 548 steps at lr 0.1). Randomise draws a seed,
  checks it in the background, and says so when the draw is stuck with dead
  neurons, which the article uses as a teaching point. The article states the
  80 % figure.
- **B. Switch to tanh.** Randomise practically always works. Costs: a new
  `tanh` op in the engine (not in the brief's op list), the gate rule becomes
  "a dimmer, 1 − a²", and the Code layer shows an op microtensor does not have.

Already decided by the measurements, whichever activation is used:
**biases start in [0, 0.5), not at 0.** With zero biases the sample (0, 0)
gives z = 0 in every first-layer neuron, exactly on the ReLU kink, where the
finite-difference check is meaningless, and ŷ = 0 = y so there is nothing to
backpropagate. This is the one deliberate departure from microtensor's init.

Milestone 1 status: 78 tests pass. Engine gradients match central differences
(eps 1e-5, relative error < 1e-6) on all 25 parameters for 9 seeds × 4
samples and for the batch loss; the matrix-form backward matches the engine
to 1e-12; the batch gradient equals the mean of the per-sample gradients.

## 11. Performance and accessibility

- Target 60 fps on an integrated GPU: under 150 draw calls, shared
  geometries and materials, pooled pulses, pixel ratio capped at 2 and lowered
  automatically if frame time stays above 20 ms. The descent tile renders
  only when something changed.
- `prefers-reduced-motion`: no camera glides, no pulses, no auto-play on
  load; stages still reveal in order.
- All controls are real buttons / inputs with labels; the live sentence is an
  `aria-live` region; colour is never the only carrier (sign also shown as
  "+" / "−" in labels).

## 12. Milestones

Each ends with tests green and headless screenshots reviewed at both sizes.

1. Engine + tests, seed search, activation decision
2. Static scene: room, network, labels, camera, view modes
3. Step-through + Train mode with loss curve and decision map
4. Inspector with Intuition / Math / Code
5. Neuron zoom (Exploded view)
6. Descent view + Verify
7. Article, legend, README, known limitations, polish
