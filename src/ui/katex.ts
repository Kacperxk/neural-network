import katex from 'katex';
import { HEX } from '../scene/theme';

/**
 * Colour macros shared by every formula: the same colours as the scene.
 * The hex codes are written without '#', which inside a macro body would be read as an argument marker.
 */
const hex = (c: string): string => c.slice(1);
export const MACROS: Record<string, string> = {
  '\\fw': `\\textcolor{${hex(HEX.fwd)}}{#1}`,
  '\\bk': `\\textcolor{${hex(HEX.back)}}{#1}`,
  '\\wt': `\\textcolor{${hex(HEX.pos)}}{#1}`,
  '\\tg': `\\textcolor{${hex(HEX.target)}}{#1}`,
};

export function tex(src: string, displayMode = true): string {
  return katex.renderToString(src, { displayMode, throwOnError: false, strict: false, macros: { ...MACROS } });
}

export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** Minimal Python colouring: comments dimmed, with the numbers inside them picked out. */
export function highlight(code: string): string {
  return escapeHtml(code)
    .split('\n')
    .map((line) => {
      const i = line.indexOf('#');
      if (i < 0) return line;
      const head = line.slice(0, i);
      const comment = line.slice(i);
      const isSource = /^# (nn|engine)\.py/.test(comment);
      const body = isSource ? comment : comment.replace(/(-?\d+\.\d+(?:e-?\d+)?|\?)/g, '<span class="nm">$1</span>');
      return `${head}<span class="${isSource ? 'src' : 'cm'}">${body}</span>`;
    })
    .join('\n');
}
