#!/usr/bin/env node
/**
 * End-to-end check of the FAHR-Face Live page with a fake webcam.
 *
 * Launches Chromium (Playwright) with a Y4M file as the camera, accepts the consent
 * note, waits for a face card with an age, exercises the Snapshot path once and
 * saves a full-page screenshot. Prints one JSON line to stdout; exit 0 on success.
 *
 * Usage:
 *   node scripts/e2e_fake_camera.mjs [--url http://127.0.0.1:8000] [--video path.y4m]
 *                                    [--out e2e-artifacts/webcam.png] [--timeout 60000]
 *
 * When --video does not exist it is built from tests/fixtures/.cache/obama.jpg with ffmpeg
 * (the portrait is downloaded from the face_recognition examples when missing).
 */
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PORTRAIT_URL = 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg';

/** Resolve Playwright from a local install, NODE_PATH (ESM ignores it) or the global npm root. */
async function loadPlaywright() {
  try {
    return await import('playwright');
  } catch (_) {
    /* fall through to CommonJS resolution, which honours NODE_PATH */
  }
  for (const base of [import.meta.url, path.join(process.cwd(), '__resolve__.js')]) {
    try {
      return createRequire(base)('playwright');
    } catch (_) {
      /* try next */
    }
  }
  const npmRoot = spawnSync('npm', ['root', '-g'], { encoding: 'utf8' }).stdout?.trim();
  if (npmRoot) {
    try {
      return createRequire(path.join(npmRoot, '__resolve__.js'))(path.join(npmRoot, 'playwright'));
    } catch (_) {
      /* give up below */
    }
  }
  throw new Error('Playwright not found: run `npm install -g playwright && npx playwright install chromium` or set NODE_PATH.');
}

function parseArgs(argv) {
  const args = {
    url: 'http://127.0.0.1:8000',
    video: path.join(REPO_ROOT, 'e2e-artifacts', 'obama.y4m'),
    out: path.join(REPO_ROOT, 'e2e-artifacts', 'webcam.png'),
    timeout: 60000,
    headless: true,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => argv[++i];
    if (arg === '--url') args.url = next();
    else if (arg === '--video') args.video = path.resolve(next());
    else if (arg === '--out') args.out = path.resolve(next());
    else if (arg === '--timeout') args.timeout = Number(next());
    else if (arg === '--headed') args.headless = false;
    else if (arg === '--help' || arg === '-h') {
      console.error('usage: e2e_fake_camera.mjs [--url URL] [--video FILE.y4m] [--out FILE.png] [--timeout MS] [--headed]');
      process.exit(2);
    } else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

async function ensurePortrait() {
  const jpg = path.join(REPO_ROOT, 'tests', 'fixtures', '.cache', 'obama.jpg');
  if (fs.existsSync(jpg)) return jpg;
  fs.mkdirSync(path.dirname(jpg), { recursive: true });
  const response = await fetch(PORTRAIT_URL);
  if (!response.ok) throw new Error(`Could not download portrait: HTTP ${response.status}`);
  fs.writeFileSync(jpg, Buffer.from(await response.arrayBuffer()));
  return jpg;
}

async function ensureVideo(videoPath) {
  if (fs.existsSync(videoPath)) return videoPath;
  const jpg = await ensurePortrait();
  fs.mkdirSync(path.dirname(videoPath), { recursive: true });
  const filter = 'scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2';
  const result = spawnSync(
    'ffmpeg',
    ['-y', '-loglevel', 'error', '-loop', '1', '-i', jpg, '-t', '4', '-r', '15', '-vf', filter, '-pix_fmt', 'yuv420p', videoPath],
    { stdio: ['ignore', 'inherit', 'inherit'] },
  );
  if (result.status !== 0) throw new Error(`ffmpeg failed to build ${videoPath} (exit ${result.status})`);
  return videoPath;
}

async function fetchHealth(url) {
  const response = await fetch(new URL('/api/health', url), { cache: 'no-store' });
  if (!response.ok) throw new Error(`GET /api/health -> HTTP ${response.status}`);
  return response.json();
}

const text = async (locator) => ((await locator.textContent()) || '').trim();

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const summary = {
    ok: false,
    faces: 0,
    ageText: null,
    riskText: null,
    detector: null,
    latencyMs: null,
    snapshotCard: null,
    consoleErrors: [],
    screenshot: args.out,
  };
  let browser;
  let page;
  try {
    const { chromium } = await loadPlaywright();
    const video = await ensureVideo(args.video);
    const health = await fetchHealth(args.url);
    const ageWeights = health && health.models && health.models.age ? health.models.age.weights : 'unknown';
    const expectAgeMissing = ageWeights === 'none' || (health.models && health.models.age && health.models.age.available === false);

    browser = await chromium.launch({
      headless: args.headless,
      args: [
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        `--use-file-for-fake-video-capture=${video}`,
      ],
    });
    const context = await browser.newContext({ viewport: { width: 1280, height: 960 }, permissions: ['camera'] });
    await context.grantPermissions(['camera'], { origin: new URL(args.url).origin });
    page = await context.newPage();
    page.setDefaultTimeout(args.timeout);
    page.on('console', (msg) => {
      if (msg.type() === 'error') summary.consoleErrors.push(msg.text());
    });
    page.on('pageerror', (err) => summary.consoleErrors.push(`pageerror: ${err.message}`));

    await page.goto(args.url, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('body[data-health="ok"], body[data-health="error"]');
    if ((await page.getAttribute('body', 'data-health')) !== 'ok') throw new Error('Page could not load /api/health');
    await page.waitForSelector('[data-testid="status-detector"]:not([data-state="loading"])');
    summary.detector = await text(page.getByTestId('status-detector').locator('.chip-value'));

    // Consent + start. The consent note opens from "Start camera"; accept it.
    const start = page.getByTestId('start-camera');
    if (await start.isEnabled()) await start.click();
    const accept = page.getByTestId('consent-accept');
    await accept.waitFor({ state: 'visible', timeout: 10000 });
    await accept.click();
    await page.waitForSelector('[data-testid="stop-camera"]:not([disabled])');

    // Wait for a card whose age is a number (or "not available" when weights are missing).
    await page.waitForFunction((missing) => {
      const card = document.querySelector('[data-testid="face-card"]');
      if (!card) return false;
      const age = card.querySelector('[data-testid="face-age"]');
      const value = age ? age.textContent.trim() : '';
      return missing ? /not available/i.test(value) : /\d/.test(value);
    }, expectAgeMissing);

    const firstCard = page.getByTestId('face-card').first();
    summary.faces = await page.getByTestId('face-card').count();
    summary.ageText = await text(firstCard.getByTestId('face-age'));
    summary.riskText = await text(firstCard.getByTestId('face-risk'));
    const latency = await text(page.getByTestId('latency'));
    summary.latencyMs = Number.parseInt(latency, 10) || null;

    // Snapshot path: toggle mode, analyse one frame, check a card is rendered from it.
    await page.getByTestId('mode-toggle').click();
    await page.waitForSelector('[data-testid="mode-toggle"][data-mode="snapshot"]');
    await page.getByTestId('analyze-snapshot').click();
    try {
      await page.waitForSelector('[data-testid="face-list"][data-source="snapshot"] [data-testid="face-card"]', { timeout: 20000 });
      await page.waitForSelector('[data-testid="back-to-live"]:not([hidden])', { timeout: 5000 });
      summary.snapshotCard = true;
    } catch (_) {
      summary.snapshotCard = false;
    }

    fs.mkdirSync(path.dirname(args.out), { recursive: true });
    await page.screenshot({ path: args.out, fullPage: true });
    summary.ok = summary.faces > 0 && summary.snapshotCard === true;
  } catch (err) {
    summary.error = err && err.message ? err.message : String(err);
    if (page) {
      try {
        fs.mkdirSync(path.dirname(args.out), { recursive: true });
        await page.screenshot({ path: args.out, fullPage: true });
      } catch (_) {
        /* ignore screenshot failure */
      }
    }
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
  if (summary.consoleErrors.length) {
    for (const line of summary.consoleErrors) console.error(`[console.error] ${line}`);
  }
  console.log(JSON.stringify(summary));
  process.exit(summary.ok ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
