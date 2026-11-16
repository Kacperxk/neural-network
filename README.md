# Backprop Room

An interactive 3D explainer of how a tiny neural network learns, with backpropagation as the centrepiece.

One network (2 → 3 → 3 → 1, ReLU, 25 parameters) learns XOR with plain gradient descent. You can:

- **step through one training iteration** in 12 stages: forward, loss, backward, update;
- **click any neuron, weight or bias** to see its chain rule with the live numbers, as Intuition, Math or Code;
- **open a neuron** into its multiply, add and ReLU steps and watch the gradient flow back through it;
- **train** the whole network and watch the loss curve and decision map;
- **verify** any gradient against a direct finite-difference measurement.

The goal is that you finish able to do a backward pass by hand on this exact example. The article under the stage walks through it.

## Run it

Requires Node 20 or newer.

```bash
npm install
```

```bash
npm run dev
```

`npm test` runs the unit tests, `npm run build` builds the static site into `dist/`.

## Can I trust the numbers?

Every number on screen comes from a small scalar autograd engine (`src/engine/value.ts`). The tests check its gradients against finite differences on all 25 parameters, and against a separate hand-written matrix version of the same equations.

## Limitations

- One fixed network, dataset and loss. No momentum, Adam, mini-batches or regularisation.
- With only three hidden neurons per layer, about 1 in 5 random starts gets stuck on dead ReLUs. The four presets always converge; **Randomise** tells you when a draw is stuck.
- Displayed numbers are rounded (2 decimals on the stage, 3 in the cards), so recomputing by hand can differ in the last digit.
- The descent view is a 2-D slice of a 25-D loss surface.
- Built for desktop. It runs on a phone, but the layout is cramped.

The article's "How accurate it is" section lists every simplification in the visuals.

## Credit

Style inspired by the work of Sael, [sael.net](https://sael.net). This is an independent project, not affiliated with or endorsed by Sael; no code, text or assets were copied.

The engine mirrors two of my earlier projects, copied into [`reference/`](reference):
[microtensor](https://github.com/Kacperxk/kacper-ml-journey/tree/main/phase0/projects/microtensor) (scalar autograd) and
[numpy_neural_net](https://github.com/Kacperxk/kacper-ml-journey/tree/main/phase0/projects/numpy_neural_net) (hand-derived backward pass with a gradient check).

Built with [Three.js](https://threejs.org), [KaTeX](https://katex.org) and [Vite](https://vite.dev).
