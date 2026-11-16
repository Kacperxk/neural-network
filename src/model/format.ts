/** One number formatter for the whole app, so the scene, the cards and the article never disagree. */

const MINUS = '−';

/** Fixed decimals; values too small to show are written in scientific form instead of a misleading 0.000. */
export function fmt(x: number, digits = 3): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? 'NaN' : x > 0 ? '∞' : `${MINUS}∞`;
  if (x === 0) return (0).toFixed(digits);
  const limit = 0.5 * 10 ** -digits;
  let s: string;
  if (Math.abs(x) < limit) {
    s = x.toExponential(1).replace('e-', 'e-').replace('e+', 'e');
  } else if (Math.abs(x) >= 1e5) {
    s = x.toExponential(2).replace('e+', 'e');
  } else {
    s = x.toFixed(digits);
  }
  return s.replace('-', MINUS);
}

/**
 * Short form for labels floating in the scene: always two decimals.
 * A tiny negative number keeps its sign (shown as -0.00), so a z just below zero still reads as "closed".
 */
export function fmt2(x: number): string {
  if (!Number.isFinite(x)) return fmt(x, 2);
  const s = x.toFixed(2);
  return (x === 0 ? '0.00' : s).replace('-', MINUS);
}

/** Always shows the sign, so colour is never the only carrier of it. */
export function fmtSigned(x: number, digits = 3): string {
  const s = fmt(x, digits);
  return x > 0 ? `+${s}` : s;
}

/** Plain ASCII minus, for code listings. */
export function fmtCode(x: number, digits = 3): string {
  return fmt(x, digits).replace(MINUS, '-');
}

/** For KaTeX: ASCII minus (KaTeX typesets it properly) and 1.2e-5 as 1.2 x 10^-5. */
export function fmtTex(x: number, digits = 3): string {
  const s = fmtCode(x, digits);
  const m = /^(-?[\d.]+)e(-?\d+)$/.exec(s);
  return m ? `${m[1]}\\times 10^{${m[2]}}` : s;
}

/** Negative numbers get brackets when they are a factor in a product. */
export function texFactor(x: number, digits = 3): string {
  return `(${fmtTex(x, digits)})`;
}

export function fmtSci(x: number): string {
  if (x === 0) return '0';
  return x.toExponential(2).replace('-', MINUS).replace('e' + MINUS, 'e' + MINUS).replace('e+', 'e');
}
