export type Sample = { x: [number, number]; y: number };

/** XOR: the output is 1 when exactly one input is 1. */
export const XOR: readonly Sample[] = [
  { x: [0, 0], y: 0 },
  { x: [0, 1], y: 1 },
  { x: [1, 0], y: 1 },
  { x: [1, 1], y: 0 },
];
