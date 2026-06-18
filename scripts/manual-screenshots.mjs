// Capture representative screenshots of each main uSTAT tab for MANUAL.md.
// Requires the dev server on :5173 with the temporary window.__store dev hook.
// Run: node scripts/manual-screenshots.mjs
import { chromium } from '/Users/yh/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'docs', 'manual', 'img');

// A small realistic clinical dataset so every variable picker is populated.
const SESSION = {
  session_id: 'manual',
  filename: 'cohort.csv',
  rows: 12,
  columns: [
    { name: 'age', dtype: 'float64', kind: 'numeric' },
    { name: 'sex', dtype: 'object', kind: 'categorical' },
    { name: 'bmi', dtype: 'float64', kind: 'numeric' },
    { name: 'ldl', dtype: 'float64', kind: 'numeric' },
    { name: 'group', dtype: 'object', kind: 'categorical' },
    { name: 'time', dtype: 'float64', kind: 'numeric' },
    { name: 'event', dtype: 'int64', kind: 'categorical' },
    { name: 'score', dtype: 'float64', kind: 'numeric' },
  ],
  preview: Array.from({ length: 12 }, (_, i) => ({
    age: 50 + i, sex: i % 2 ? 'M' : 'F', bmi: 22 + (i % 5),
    ldl: 90 + i * 4, group: i % 2 ? 'A' : 'B', time: 100 + i * 10,
    event: i % 3 === 0 ? 1 : 0, score: 0.2 + i * 0.05,
  })),
};

// [filename, activeTab, optional sub-tab button text to click after switching]
const SHOTS = [
  ['01-data', 'data'],
  ['02-summary', 'summary'],
  ['03-table1', 'table1'],
  ['04-tests', 'tests'],
  ['05-correlation', 'correlation'],
  ['06-roc', 'roc'],
  ['07-models-regression', 'models', 'Regression'],
  ['08-models-survival', 'models', 'Survival Advanced'],
  ['09-psm', 'psm'],
  ['10-iptw', 'iptw'],
  ['11-causal', 'causal'],
  ['12-dca', 'dca'],
  ['13-meta', 'meta'],
  ['14-missing', 'missing'],
  ['15-visual', 'visual'],
  ['16-power', 'power'],
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({
  // The bundled headless_shell is blocked by macOS system policy in this
  // sandbox (spawn -88); the full Chrome-for-Testing build is allowed.
  executablePath: '/Users/yh/Library/Caches/ms-playwright/chromium-1228/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
  headless: true,
});
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });

for (const [file, tab, subText] of SHOTS) {
  await page.evaluate(({ session, tab }) => {
    window.__store.setState({ session, activeTab: tab });
  }, { session: SESSION, tab });
  await sleep(500);
  if (subText) {
    // Click the in-panel sub-tab button (combos read their sub from state at
    // mount, so localStorage seeding alone doesn't switch it).
    const btn = page.locator(`button:has-text("${subText}")`).first();
    if (await btn.count()) { await btn.click(); await sleep(300); }
  }
  await sleep(700);
  await page.screenshot({ path: join(OUT, `${file}.png`) });
  console.log('captured', file);
}

await browser.close();
console.log('done →', OUT);
