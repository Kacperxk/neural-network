/**
 * Initialisations that are verified to converge (tests/presets.test.ts re-checks each one).
 * Found with scripts/seed-search.ts: plain full-batch GD, mean squared error over the 4 XOR points.
 * `steps` is the number of GD steps to reach mean loss < 1e-3 at the preset's learning rate.
 */
export type Preset = { name: string; seed: number; lr: number; steps: number; note: string };

export const TARGET_LOSS = 1e-3;

export const PRESETS: Preset[] = [
  { name: 'Mixed gates', seed: 33, lr: 0.1, steps: 133, note: 'Some ReLUs start open, some closed.' },
  { name: 'Far off', seed: 122, lr: 0.1, steps: 294, note: 'Starts with a large loss.' },
  { name: 'Slow start', seed: 43, lr: 0.1, steps: 548, note: 'Sits on a plateau before it finds the way down.' },
  { name: 'All open', seed: 235, lr: 0.1, steps: 187, note: 'Every ReLU is open on every sample at the start.' },
];

export const DEFAULT_PRESET = PRESETS[0];
