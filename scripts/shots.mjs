/**
 * Headless screenshots of the app in a list of states, at 1280x720 and 1920x1080.
 *   node scripts/shots.mjs            all shots
 *   node scripts/shots.mjs loss exp   only shots whose name contains one of the words
 *   --no-build  reuse ./dist   --mobile  390x844 with touch   --motion  animations on
 * Output: ./shots/<name>-<width>.png, plus a report of console errors and overlapping panels.
 */
import { mkdirSync } from 'node:fs';
import { chromium } from 'playwright';
import { build, preview } from 'vite';

const SIZES = [
  [1280, 720],
  [1920, 1080],
];

/** Each shot: a name and a function body run in the page with `store` and `stage` in scope. */
const SHOTS = [
  ['idle', ``],
  ['fwd', `store.goto(4);`],
  ['loss', `store.goto(5); store.select({kind:'neuron', col:3, i:0});`],
  ['delta1', `store.goto(10); store.select({kind:'neuron', col:1, i:0});`],
  ['grad-weight', `store.goto(11); store.select({kind:'weight', l:1, j:0, k:0});`],
  ['update', `store.goto(12); store.select({kind:'weight', l:2, j:0, k:0}); store.set({showWeightLabels:true});`],
  ['verify', `store.goto(7); store.select({kind:'weight', l:2, j:0, k:1}); store.verify({kind:'w', l:2, j:0, k:1});`],
  ['train', `store.setMode('train'); store.trainStep(60); store.select({kind:'weight', l:0, j:0, k:0});`],
  ['train-done', `store.setMode('train'); store.trainStep(200);`],
  ['exploded', `store.explode(0, 0); for (let i = 0; i < 5; i++) store.explodedNext();`],
  ['exploded-out', `store.explode(2, 0); for (let i = 0; i < 4; i++) store.explodedNext();`],
  ['studio', `store.goto(9); store.setView('studio');`],
  ['legend', `store.set({legendOpen: true});`],
  ['descent-big', `store.setLr(0.9); store.set({descent: {...store.s.descent, big: true}});`],
];

const filters = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const mobile = process.argv.includes('--mobile');
const motion = process.argv.includes('--motion');

// Screenshots are taken of the production build, served by `vite preview`: that is what ships.
if (!process.argv.includes('--no-build')) await build({ logLevel: 'error' });
const server = await preview({ preview: { port: 5199, strictPort: false }, logLevel: 'error' });
const url = server.resolvedUrls.local[0];
mkdirSync('shots', { recursive: true });

const launch = () =>
  chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl'] });
const browser = await launch();
const problems = [];

for (const [w, h] of mobile ? [[390, 844]] : SIZES) {
  const ctx = await browser.newContext({
    viewport: { width: w, height: h },
    reducedMotion: motion ? 'no-preference' : 'reduce',
    hasTouch: mobile,
  });
  const page = await ctx.newPage();
  page.on('console', (m) => {
    if (m.type() === 'error') problems.push(`[${w}] console: ${m.text()}`);
  });
  page.on('pageerror', (e) => problems.push(`[${w}] pageerror: ${e.message}`));

  for (const [name, body] of SHOTS) {
    if (filters.length && !filters.some((f) => name.includes(f))) continue;
    await page.goto(url);
    await page.waitForFunction(() => window.backprop && document.getElementById('loading').classList.contains('done'));
    await page.evaluate(`(() => { const { store, stage } = window.backprop; ${body} })()`);
    await page.waitForTimeout(motion ? 500 : 900);
    await page.screenshot({ path: `shots/${name}-${w}.png` });

    // Overlap check between the fixed panels of the stage.
    const overlaps = await page.evaluate(() => {
      const ids = ['.brand', '.centre', '#console', '#inspector', '#dock', '#readon'];
      const rects = ids
        .map((sel) => [sel, document.querySelector(sel)])
        .filter(([, el]) => el && !el.hidden && getComputedStyle(el).display !== 'none')
        .map(([sel, el]) => {
          // Use the union of the children: containers like .centre are wider than what they draw.
          const scrolls = getComputedStyle(el).overflowY === 'auto';
          const kids = scrolls ? [] : [...el.children].filter((c) => !c.hidden && getComputedStyle(c).display !== 'none');
          const rs = (kids.length ? kids : [el]).map((k) => k.getBoundingClientRect()).filter((r) => r.width > 0);
          return [sel, rs];
        });
      const out = [];
      for (let i = 0; i < rects.length; i++)
        for (let j = i + 1; j < rects.length; j++)
          for (const a of rects[i][1])
            for (const b of rects[j][1])
              if (a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1)
                out.push(`${rects[i][0]} overlaps ${rects[j][0]}`);
      const vw = innerWidth;
      const vh = innerHeight;
      for (const [sel, rs] of rects)
        for (const r of rs) if (r.right > vw + 1 || r.bottom > vh + 1 || r.left < -1 || r.top < -1) out.push(`${sel} leaves the viewport`);
      return [...new Set(out)];
    });
    for (const o of overlaps) problems.push(`[${w}] ${name}: ${o}`);
    console.log(`shot ${name}-${w}`);
  }
  await ctx.close();

  // The article, one viewport at a time. It gets a browser of its own: headless Chromium with software GL
  // stops loading new pages after a run of screenshots taken with the WebGL canvas scrolled off screen.
  if (!filters.length || filters.includes('article')) {
    const ab = await launch();
    const actx = await ab.newContext({ viewport: { width: w, height: h }, reducedMotion: 'reduce' });
    const ap = await actx.newPage();
    ap.on('pageerror', (e) => problems.push(`[${w}] article pageerror: ${e.message}`));
    await ap.goto(url);
    await ap.waitForFunction(() => window.backprop && document.getElementById('loading').classList.contains('done'));
    const total = await ap.evaluate(() => document.documentElement.scrollHeight);
    let n = 0;
    for (let y = h; y < total && n < 24; y += h - 60) {
      await ap.evaluate((yy) => window.scrollTo(0, yy), y);
      await ap.waitForTimeout(300);
      await ap.screenshot({ path: `shots/article${String(n++).padStart(2, '0')}-${w}.png` });
    }
    if (await ap.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)) problems.push(`[${w}] article scrolls sideways`);
    const bad = await ap.evaluate(() => document.querySelectorAll('#article .katex-error').length);
    if (bad) problems.push(`[${w}] ${bad} KaTeX errors in the article`);
    const wideEq = await ap.evaluate(() => [...document.querySelectorAll('#article .eq')].filter((e) => e.scrollWidth > e.clientWidth + 2).length);
    if (wideEq) problems.push(`[${w}] ${wideEq} article equations are wider than the column (they scroll sideways)`);
    console.log(`article-${w}: ${n} slices`);
    await ab.close();
  }
}

await browser.close();
await new Promise((resolve) => server.httpServer.close(resolve));
console.log(problems.length ? `\nPROBLEMS:\n${problems.join('\n')}` : '\nNo console errors, no overlapping panels.');
